(omega-dev-metadata)=

## Metadata

The MetaData classes in OMEGA manage and manipulate metadata attached to
all fields as well as global metadata that apply to the code or current
simulation/experiment. Additional metadata is required to define dimensions
for array fields. Metadata groups create a shortcut for referring to sets of
fields that are commonly grouped together and reduce the size of contents
lists in IOStreams. There are three classes used to create, retrieve and
manipulate metadata. To use these classes, you must include the `MetaData.h`
header file.

### MetaData Class

The MetaData class is the main class for defining fields and creating
metadata. The class stores the name of the field, the dimension information
(for array fields) and a `std::map` of (name, value) pairs for all metadata.

For most array-based fields in OMEGA, there is a create function
to define a field and ensure that all required metadata is included:
``` c++
   auto MyMeta = MetaData::create(
          const std::string Name,        ///< [in] Name of variable/field
          const std::string Description, ///< [in] long Name or description
          const std::string Units,       ///< [in] units
          const std::string StdName,     ///< [in] CF standard Name
          const std::any ValidMin,       ///< [in] min valid field value
          const std::any ValidMax,       ///< [in] max valid field value
          const std::any FillValue,      ///< [in] scalar for undefined entries
          const int NumDims,             ///< [in] number of dimensions
          const std::vector<std::string> Dimensions ///< dim names for each dim
   );
```
MyMeta is actually a `std::shared_ptr` to the defined metadata so all member
functions are referenced using `MyMeta->functionName`.
If a field of that name already exists, an error message is generated and
the pointer is set to nullptr.
For global metadata not attached to a particular field, we provide two standard
field names `CodeMeta` and `SimMeta`. These and other non-array fields
can be created using the functions:
```c++
   auto MyMeta = MetaData::create(Name);
```
or
```c++
   auto Data2 = MetaData::create(
                "MyVar2",
                {
                    std::make_pair("Meta1", 1),
                    std::make_pair("Meta2", true),
                    std::make_pair("Meta3", "MyValue3"),
                }
            );
```
The second form allows multiple key-value pairs to be added at once during
creation. All create methods add the field and metadata to a list of all
defined fields and also perform checks to ensure no duplicate fields are
created. Fields can later be retrieved by name.

Once defined, additional metadata pairs can be added or removed using:
```c++
   int Err = MyMeta->addEntry(Name, Value);

   int Err = MyMeta->removeEntry(Name);
```
Note that internally the `std::any` type is used to simplify the storage
and reduce the number of interfaces. So the Value must be a standard supported
data type (I4, I8, R4, R8, bool, std::string). When retrieving the data,
the `std::any` is coerced to the appropriate data type.

There are multiple mechanisms for retrieving metadata. To retrieve a pointer
to the full metadata instance associated with a field, use:
```c++
   auto MyMeta = get(Name);
```
Individual metadata entries can then be retrieved using:
```c++
   int Err = MyMeta->getEntry("EntryName", Value);
```
The getEntry function is overloaded for all supported OMEGA data types
(I4, I8, R4, R8, bool, std::string) and the `getEntry` function will attempt
to coerce the entry to the type associated with Value in the argument list.
The entire `std::map` of all entries can be retrieved with:
```c++
   std::map<std::string, std::any> *AllEntries = MyMeta->getAllEntries();
```

For array fields, the dimension information can also be extracted from
the MetaData using:
```c++
   int NumDims = MyMeta->getNumDims();

   std::vector<std::string> DimNames(NumDims);
   int Err = getDimNames(DimNames);
```
The dimension name vector stores the dimension names in the same index order
of the array itself. Once the list of dimension names has been retrieved
the dimension query functions listed below can be used to get the length.

To check whether a field has been defined or whether an entry already
exists, use the `has` functions:
```c++
   bool MetaExists = MetaData::has(FieldName);

   bool EntryExists = MyMeta->hasEntry(Name);
```
These functions can be used within a conditional.

Finally, there are functions to remove a defined field:
```c++
   int Err = MyMeta->destroy(FieldName);
```
or remove all defined fields and metadata (typically before the code exits):
```c++
   MetaData::clear();
```

### MetaGroup class

To group metadata fields together, we provide a MetaGroup class. A group
can be defined as a list of member fields. Fields can be members of more
than one group. The group name simply provides a shortcut for referring to
fields that are commonly grouped together. A class instance contains the
name of the group, a list of fields stored as a sorted list using the
`std::set` container. The class also tracks and manages a list of all groups
that have been defined.

To create a group, first create an empty group with a given name using:
```c++
   auto MyGroup = MetaGroup::create(Name);
```
As in the MetaData class, a shared pointer to the group is returned so member
functions are called using the form `MyGroup->function()`

Once the group has been created, add fields by name using:
```c++
   int Err = MyGroup->addField(Name);
```
The field with the given name should have already been defined before adding
to a group.
If necessary, fields can later be removed from a group using:
```c++
   int Err = MyGroup->removeField(Name);
```

The existence of a group or a field within a group can be checked using:
```c++
   bool GroupExists = MetaGroup::has(Name);

   bool FieldIsInGroup = MyGroup->hasField(Name);
```
A list of names for all fields in a group is available with:
```c++
   std::set<std::string> FieldList = MyGroup->getFieldList();
```

The group can be later retrieved by name using:
```c++
   auto MyGroup = MetaGroup::get(Name);
```
and the field metadata for any group member can be retrieved using:
```
   auto FieldMeta = MyGroup->getField(Name);
```

Groups can be removed using:
```c++
   int Err = destroy(Name);
```

### MetaDim Class

The MetaDim class stores the name and length of an array dimension. The length
refers to the global (unpartitioned) length. Dimension metadata is required to
properly define arrays. As in the other metadata classes, a list of all
defined dimensions is maintained so that they can be retrieved by name later.

Creating a dimension is a simple call:
```c++
   auto MyDim1 = MetaDim::create(Name, Length);
```
Unlike other metadata create functions, if a dimension has already been
defined with the same name and length, the creation returns the shared pointer
to the previously defined dimension and no new dimension is created.
The existence of a dimension can be queried using:
```c++
   bool DimExists = MetaDim::has(Name);
```

A defined dimension can be retrieved by name using:
```c++
   auto MyDim = MetaDim::get(Name);
```
There are two methods to retrieve the length of a dimension. If the dimension
has already been retrieved, the member function can be used:
```c++
   I4 Length = MyDim->getLength();
```
but the length can be retrieved without getting the dimension pointer using:
```c++
   I4 Length = MetaDim::getDimLength(Name);
```

In some cases, it is useful to access all the dimensions (eg to write the
dimension metadata to a file). For this purpose, an iterator is available
and can be used in a loop like:
```c++
   for (auto It = MetaDim::begin(); It != MetaDim::end(); ++It) {
      std::string Name = It->first;
      I4 Length = MetaDim::getDimLength(Name);
      // Do stuff with name, length
   }
```
The total number of defined dimensions can be obtained with:
```c++
   int TotalDims = MetaDim::getNumDefinedDims();
```

Finally, dimensions can be deleted using:
```c++
   int Err = MetaDim::destroy(Name);
```
and all dimensions can be removed with:
```c++
   MetaDim::clear();
```
