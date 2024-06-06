//===-- infra/MetaData.cpp - Omega MetaData implementation ------*- C++ -*-===//
//
// \file
// \brief Implements Omega metadata functions
//
// This implements Omega metadata classes that define and store metadata
// associated with Omega fields. These are then used to output metadata for
// self-describing files. Fields can also be grouped together to provide a
// more compact way of referring to a set of fields that are commonly used
// together.
//
//===----------------------------------------------------------------------===//

#include "MetaData.h"
#include "Logging.h"

namespace OMEGA {

// Create static class members that store instantiations of metadata
std::map<std::string, std::shared_ptr<MetaData>> MetaData::AllFields;
std::map<std::string, std::shared_ptr<MetaDim>> MetaDim::AllDims;
std::map<std::string, std::shared_ptr<MetaGroup>> MetaGroup::AllGroups;

//----------------------------------------------------------------------------//
// Checks for the existence of a field with the given name
bool MetaData::has(const std::string Name // [in] Name of field
) {
   return (AllFields.find(Name) != AllFields.end());
}

//----------------------------------------------------------------------------//
// Creates an empty metadata container for a field with input name
// Generates an error message if field already exists.
std::shared_ptr<MetaData>
MetaData::create(const std::string Name // [in] Name of field
) {

   if (has(Name)) {
      LOG_ERROR("Failed to create a field instance because {} already exists.",
                Name);
      return nullptr;
   }

   auto Data = std::make_shared<MetaData>();

   Data->FieldName = Name;
   AllFields[Name] = Data;

   return Data;

} // end create for empty metadata

//----------------------------------------------------------------------------//
// Creates the metadata for an array field. This is the preferred create
// interface for most fields in Omega. It enforces a list of required
// metadata. Note that if input parameters don't exist
// (eg stdName) or don't make sense (eg min/max or fill) for a
// given field, empty or 0 entries can be provided. Similarly
// for scalars, NumDims can be 0 and an empty Dimensions vector can be
// supplied.
std::shared_ptr<MetaData> MetaData::create(
    const std::string Name,        // [in] Name of variable/field
    const std::string Description, // [in] long Name or description
    const std::string Units,       // [in] units
    const std::string StdName,     // [in] CF standard Name
    const std::any ValidMin,       // [in] min valid field value
    const std::any ValidMax,       // [in] max valid field value
    const std::any FillValue,      // [in] scalar for undefined entries
    const int NumDims,             // [in] number of dimensions
    const std::vector<std::string> Dimensions // [in] dim names for each dim
) {

   auto Data = MetaData::create(Name); // create and names an empty instance

   // Fill metadata if creation was successful
   if (Data) { // creation successful

      Data->NDims    = NumDims;
      Data->DimNames = Dimensions;

      Data->addEntry("Description", Description);
      Data->addEntry("Units", Units);
      Data->addEntry("StdName", StdName);
      Data->addEntry("ValidMin", ValidMin);
      Data->addEntry("ValidMax", ValidMax);
      Data->addEntry("FillValue", FillValue);

   } else { // creation not successful, return with error
      LOG_ERROR("Metadata create failed for field {} ", Name);
   }

   return Data;
}

//----------------------------------------------------------------------------//
// Creates metadata for field with given name and list of (name,value)
// pairs
std::shared_ptr<MetaData> MetaData::create(
    const std::string Name, // [in] name of field to create
    const std::initializer_list<std::pair<std::string, std::any>> &MetaPairs) {

   auto Data = create(Name); // creates and names an empty class instance

   // Fill metadata if creation was successful
   if (Data) { // creation successful

      Data->NDims = 0; // assume this is not an array field
      for (const auto &MetaPair : MetaPairs) {
         Data->addEntry(MetaPair.first, MetaPair.second);
      }

   } else { // creation not successful, return with error
      LOG_ERROR("Metadata create failed for field {} ", Name);
   }

   return Data;
}

//----------------------------------------------------------------------------//
// Removes the metadata associated with a field of a given name
int MetaData::destroy(const std::string Name // [in] Name of field to remove
) {
   int RetVal = 0;

   if (!has(Name)) {
      LOG_ERROR("Failed to destroy metadata for the field {}: "
                "field does not exist.",
                Name);
      RetVal = -1;

   } else {
      if (AllFields.erase(Name) != 1) {
         LOG_ERROR("Unknown error erasing metadata for field {} ", Name);
         RetVal = -2;
      }
   }

   return RetVal;
} // end destroy for MetaData

//----------------------------------------------------------------------------//
// Removes all defined MetaData
void MetaData::clear() { AllFields.clear(); }

//----------------------------------------------------------------------------//
// Retrieves all metadata for a field with a given name
std::shared_ptr<MetaData>
MetaData::get(const std::string Name // [in] Name of field
) {
   if (!has(Name)) {
      LOG_ERROR("Failed to retrieve metadata for field {}: "
                "Field with that name does not exist",
                Name);
      return nullptr;
   }

   return AllFields[Name];
}

//----------------------------------------------------------------------------//
// Checks for the existence of a metadata entry with the given name
bool MetaData::hasEntry(const std::string Name // [in] Name of metadata
) {
   return (MetaMap.find(Name) != MetaMap.end());
}

//----------------------------------------------------------------------------//
// Adds a metadata entry with the (name,value) pair
int MetaData::addEntry(const std::string Name, // [in] Name of new metadata
                       const std::any Value    // [in] Value of new metadata
) {
   int RetVal = 0;

   if (hasEntry(Name)) {
      LOG_ERROR("Failed to add the metadata {} to field {} because the field "
                "already has an entry with that name.",
                Name, FieldName);

      RetVal = -1;

   } else {
      MetaMap[Name] = Value;
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Removes a metadata entry with the given name
int MetaData::removeEntry(const std::string Name // [in] Name of metadata to rm
) {

   int RetVal = 0;

   if (!hasEntry(Name)) {
      LOG_ERROR("Failed to remove the metadata {} from field {} because an "
                "entry with that name does not exist.",
                Name, FieldName);
      RetVal = -1;

   } else {
      if (MetaMap.erase(Name) != 1) {
         LOG_ERROR("Unknown error trying to erase metadata {} from field {}",
                   Name, FieldName);
         RetVal = -2;
      }
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Returns the number of dimensions for an array field
int MetaData::getNumDims() { return NDims; }

//----------------------------------------------------------------------------//
// Returns a vector of dimension names associated with each dimension
// of an array field. Returns an error code.
int MetaData::getDimNames(
    std::vector<std::string> &Dimensions // [out] list of dimensions
) {
   int RetVal = 0;

   if (NDims > 0) { // This field is an array

      Dimensions.resize(NDims); // make sure the vector has correct size
      // Copy dimension names into result
      for (int IDim = 0; IDim < NDims; ++IDim) {
         Dimensions[IDim] = DimNames[IDim];
      }

   } else { // not an array so clear the vector of names
      Dimensions.clear();
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Retrieves the value of the metadata associated with a given name
// This specific version of the overloaded interface coerces the value
// to an I4 type
int MetaData::getEntry(const std::string Name, // [in] Name of metadata to get
                       I4 &Value               // [out] I4 Value of metadata
) {
   int RetVal = 0;

   if (!hasEntry(Name)) {
      LOG_ERROR("Metadata {} does not exist for field {}.", Name, FieldName);
      RetVal = -1;

   } else {
      Value = std::any_cast<I4>(MetaMap[Name]);
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Retrieves the value of the metadata associated with a given name
// This specific version of the overloaded interface coerces the value
// to an I8 type
int MetaData::getEntry(const std::string Name, // [in] Name of metadata to get
                       I8 &Value               // [out] I8 Value of metadata
) {
   int RetVal = 0;

   if (!hasEntry(Name)) {
      LOG_ERROR("Metadata {} does not exist for field {}.", Name, FieldName);
      RetVal = -1;

   } else {
      Value = std::any_cast<I8>(MetaMap[Name]);
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Retrieves the value of the metadata associated with a given name
// This specific version of the overloaded interface coerces the value
// to an R4 type
int MetaData::getEntry(const std::string Name, // [in] Name of metadata to get
                       R4 &Value               // [out] R4 Value of metadata
) {
   int RetVal = 0;

   if (!hasEntry(Name)) {
      LOG_ERROR("Metadata {} does not exist for field {}.", Name, FieldName);
      RetVal = -1;

   } else {
      Value = std::any_cast<R4>(MetaMap[Name]);
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Retrieves the value of the metadata associated with a given name
// This specific version of the overloaded interface coerces the value
// to an R8 type
int MetaData::getEntry(const std::string Name, // [in] Name of metadata to get
                       R8 &Value               // [out] R8 Value of metadata
) {
   int RetVal = 0;

   if (!hasEntry(Name)) {
      LOG_ERROR("Metadata {} does not exist for field {}.", Name, FieldName);
      RetVal = -1;

   } else {
      Value = std::any_cast<R8>(MetaMap[Name]);
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Retrieves the value of the metadata associated with a given name
// This specific version of the overloaded interface coerces the value
// to a bool type
int MetaData::getEntry(const std::string Name, // [in] Name of metadata to get
                       bool &Value             // [out] Bool Value of metadata
) {
   int RetVal = 0;

   if (!hasEntry(Name)) {
      LOG_ERROR("Metadata {} does not exist for field {}.", Name, FieldName);
      RetVal = -1;

   } else {
      Value = std::any_cast<bool>(MetaMap[Name]);
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Retrieves the value of the metadata associated with a given name
// This specific version of the overloaded interface coerces the value
// to an string type
int MetaData::getEntry(const std::string Name, // [in] Name of metadata to get
                       std::string &Value      // [out] String Value of metadata
) {
   int RetVal = 0;

   if (!hasEntry(Name)) {
      LOG_ERROR("Metadata {} does not exist for field {}.", Name, FieldName);
      RetVal = -1;

   } else {
      Value = std::any_cast<std::string>(MetaMap[Name]);
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Returns the pointer to the metadata map of all entries
std::map<std::string, std::any> *MetaData::getAllEntries() { return &MetaMap; }

//----------------------------------------------------------------------------//
// Determines whether the dimension exists or has been defined
bool MetaDim::has(const std::string Name // [in] Name of dimension
) {
   return (AllDims.find(Name) != AllDims.end());
}

//----------------------------------------------------------------------------//
// Creates dimension metadata and adds it to list of all defined dimensions
// If a dimension with the same name and same length has already been
// defined, this function returns the previously defined dimension.
std::shared_ptr<MetaDim>
MetaDim::create(const std::string Name, // [in] Name of dimension
                const I4 Length         // [in] Length of dimension
) {

   auto Dim = std::make_shared<MetaDim>(); // create empty dim

   if (has(Name)) { // Dimension already exists

      Dim = get(Name); // retrieve previously defined dim
      // If already-defined dim has a different length, write error msg
      if (Dim->Length != Length) {
         LOG_ERROR("Attempt to create dimension {} but a dimension with"
                   " that name already exists with different length",
                   Name);
         Dim = nullptr;
      }

   } else { // a new dimension, fill data members

      Dim->DimName  = Name;
      Dim->Length   = Length;
      AllDims[Name] = Dim;
   }

   return Dim;
}

//----------------------------------------------------------------------------//
// Destroys dimension metadata and removes it from the list
int MetaDim::destroy(const std::string Name // [in] Name of dimension
) {
   int RetVal = 0;

   if (!has(Name)) {
      LOG_ERROR("Failed to destroy the dimension {} because "
                "the dimension name does not exist.",
                Name);
      RetVal = -1;

   } else {
      if (AllDims.erase(Name) != 1) {
         LOG_ERROR("Unknown error erasing Dimension {}", Name);
         RetVal = -1;
      }
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Removes all defined dimension info
void MetaDim::clear() {
   AllDims.clear(); // uses map clear function to remove all entries
}

//----------------------------------------------------------------------------//
// Retrieves dimension metadata by name
std::shared_ptr<MetaDim>
MetaDim::get(const std::string Name // [in] Name of dimension
) {
   if (!has(Name)) {
      LOG_ERROR("Failed to retrieve dimension {} because it does not exist"
                " or has not net been defined",
                Name);
      return nullptr;
   }

   return AllDims[Name];
}

//----------------------------------------------------------------------------//
// Retrieves the length of the dimension
I4 MetaDim::getLength() { return Length; }

//----------------------------------------------------------------------------//
// Retrieves the length of the dimension by name
I4 MetaDim::getDimLength(const std::string &Name // [in] Name of dimension
) {
   I4 Length;

   if (has(Name)) {
      std::shared_ptr<MetaDim> ThisDim = AllDims[Name];
      Length                           = ThisDim->Length;

   } else {
      LOG_ERROR("Cannot get length of dimension {}: "
                "dimension does not exist or has not been defined",
                Name);
      Length = -1;
   }

   return Length;
}

//----------------------------------------------------------------------------//
// Retrieves the number of currently defined dimensions
int MetaDim::getNumDefinedDims() { return AllDims.size(); }

//----------------------------------------------------------------------------//
// Returns an iterator to the first dimension stored in AllDims
MetaDim::Iter MetaDim::begin() { return MetaDim::AllDims.begin(); }

//----------------------------------------------------------------------------//
// Returns an iterator to the last dimension stored in AllDims
MetaDim::Iter MetaDim::end() { return MetaDim::AllDims.end(); }

//----------------------------------------------------------------------------//
// Determines whether a group of a given name exists
bool MetaGroup::has(const std::string Name // [in] Name of group
) {
   return (AllGroups.find(Name) != AllGroups.end());
}

//----------------------------------------------------------------------------//
// Creates an empty metadata group with a given name
std::shared_ptr<MetaGroup>
MetaGroup::create(const std::string Name // [in] Name of group
) {
   if (has(Name)) {
      LOG_ERROR("Attempt to create a metadata group {} that already exists.",
                Name);
      return nullptr;

   } else {
      auto Group = std::make_shared<MetaGroup>();

      Group->GrpName  = Name;
      AllGroups[Name] = Group;

      return Group;
   }
}

//----------------------------------------------------------------------------//
// Removes a metadata group
int MetaGroup::destroy(const std::string Name // [in] Name of group
) {
   int RetVal = 0;

   if (!has(Name)) {
      LOG_ERROR("Failed to destroy the metadata group {}: group not found",
                Name);
      RetVal = -1;

   } else {
      if (AllGroups.erase(Name) != 1) {
         LOG_ERROR("Unknown error trying to erase metadata group {}.", Name);
         RetVal = -2;
      }
   }

   return RetVal;
}

//----------------------------------------------------------------------------//
// Retrieves a pointer to a metadata group by name
std::shared_ptr<MetaGroup>
MetaGroup::get(const std::string Name // [in] Name of group to retrieve
) {
   if (!has(Name)) {
      LOG_ERROR("Failed to retrieve MetaGroup {}: group does not exist.", Name);
      return nullptr;
   }

   return AllGroups[Name];
}

//----------------------------------------------------------------------------//
// Determines whether a field of a given name exists in the group
bool MetaGroup::hasField(const std::string FieldName // [in] Name of field
) {
   return (Fields.find(FieldName) != Fields.end());
}

//----------------------------------------------------------------------------//
// Adds a field to the group
int MetaGroup::addField(const std::string FieldName // [in] Name of field to add
) {
   int RetVal = 0;

   // Make sure the field metadata has already been defined
   if (not MetaData::has(FieldName)) {
      LOG_ERROR("Cannot add the field {} to Group {}: Field not defined",
                FieldName, GrpName);
      RetVal = -2; // The field name does not exist.
   }

   // Add field name to the list of fields. If this is a duplicate, the
   // insert function knows not to repeat an entry so nothing happens.
   Fields.insert(FieldName);

   return RetVal;
}

//----------------------------------------------------------------------------//
// Retrieves a field's metadata from a group
std::shared_ptr<MetaData>
MetaGroup::getField(const std::string FieldName // [in] Name of field to add
) {
   if (!hasField(FieldName)) {
      LOG_ERROR("Failed to get field {} from group {}: field not in group.",
                FieldName, GrpName);
      return nullptr;
   }

   return MetaData::get(FieldName);
}

//----------------------------------------------------------------------------//
// Removes a field from a group
int MetaGroup::removeField(
    const std::string FieldName // [in] Name of field to remove
) {
   int RetVal = 0;

   if (!hasField(FieldName)) {
      LOG_ERROR("Failed to remove field {} from group {} "
                "because the field is not a group member",
                FieldName, GrpName);
      RetVal = -1;

   } else {
      if (Fields.erase(FieldName) != 1) {
         LOG_ERROR("Unknown error erasing field {} from group {}.", FieldName,
                   GrpName);
         RetVal = -2;
      }
   }

   return RetVal;
} // end removeField from group

//----------------------------------------------------------------------------//
// Returns a list of all fields in a group - the list is a copy of the group
// field list so the internal list is not affected
std::set<std::string> MetaGroup::getFieldList() {

   std::set<std::string> List;
   for (auto It = Fields.begin(); It != Fields.end(); ++It) {
      List.insert(*It);
   }

   return List;
}

} // Namespace OMEGA
