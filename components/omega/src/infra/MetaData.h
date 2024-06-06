#ifndef OMEGA_METADATA_H
#define OMEGA_METADATA_H
//===-- infra/MetaData.h - Omega metadata classes ---------------*- C++ -*-===//
//
/// \file
/// \brief Defines metadata classes
///
/// This header defines classes and functions for defining and storing metadata
/// associated with Omega fields. These are then used to output metadata for
/// self-describing files. Fields can also be combined into groups to provide a
/// more compact way of referring to a set of fields that are commonly used
/// together.
//===----------------------------------------------------------------------===//

#include "DataTypes.h"
#include <any>
#include <map>
#include <set>

namespace OMEGA {

static const std::string CodeMeta{"code"};      ///< name for code metadata
static const std::string SimMeta{"simulation"}; ///< name for sim metatdata

//------------------------------------------------------------------------------
/// The MetaDim class stores dimension information. The information is stored
/// in a (name,length) map pair and iterators are provided for looping through
/// all defined dimensions.
class MetaDim {

 private:
   /// Name of the dimension
   std::string DimName;

   /// The length of the dimension. Use 0 for unlimited length.
   I4 Length;

   /// Store and maintain all defined dimensions
   static std::map<std::string, std::shared_ptr<MetaDim>> AllDims;

 public:
   /// Creates dimension metadata and adds it to list of all defined dimensions
   /// If a dimension with the same name and same length has already been
   /// defined, this function returns the previously defined dimension.
   static std::shared_ptr<MetaDim>
   create(const std::string Name, ///< [in] Name of dimension
          const I4 Length         ///< [in] Length (global) of dimension
   );

   /// Destroys dimension metadata and removes it from the list
   static int destroy(const std::string Name ///< [in] Name of dimension
   );

   /// Retrieves dimension metadata by name
   static std::shared_ptr<MetaDim>
   get(const std::string Name ///< [in] Name of dimension
   );

   /// Determines whether the dimension exist or has been defined
   static bool has(const std::string Name ///< [in] Name of dimension
   );

   /// Retrieves the length of the dimension
   I4 getLength();

   /// Retrieves the length of the dimension by name
   static I4 getDimLength(const std::string &Name ///< [in] Name of dimension
   );

   /// Retrieves the number of currently defined dimensions
   static int getNumDefinedDims();

   /// An iterator can be used to loop through all defined dimensions
   using Iter = std::map<std::string, std::shared_ptr<MetaDim>>::iterator;

   /// Returns an iterator to the first dimension stored in AllDims
   static Iter begin();

   /// Returns an iterator to the last dimension stored in AllDims
   static Iter end();

   /// Removes all defined dimension info
   static void clear();
};

//------------------------------------------------------------------------------
/// The MetaData class stores field metadata in a map of (string,value) pairs
/// Dimension information is also stored here for array fields since it is
/// treated differently in file metadata and I/O ops
class MetaData {
 protected:
   /// Name of field that this MetaData describes
   std::string FieldName;

   /// Most metadata stored in map
   std::map<std::string, std::any> MetaMap;

   /// Number of dimensions for the field (0 if scalar field or global metadata)
   int NDims;

   /// Dimension names for retrieval of dimension info
   /// These must be in the same index order as the stored data
   std::vector<std::string> DimNames;

   /// Store and maintain metadata for all defined fields in this map container
   static std::map<std::string, std::shared_ptr<MetaData>> AllFields;

 public:
   /// Creates the metadata for an array field. This is the preferred
   /// interface for most fields in Omega. It enforces a list of required
   /// metadata. Note that if input parameters don't exist
   /// (eg stdName) or don't make sense (eg min/max or fill) for a
   /// given field, empty or 0 entries can be provided. Similarly
   /// for scalars, NumDims can be 0 and an emtpy MetaDim vector can be
   /// supplied.
   static std::shared_ptr<MetaData>
   create(const std::string Name,        ///< [in] Name of variable/field
          const std::string Description, ///< [in] long Name or description
          const std::string Units,       ///< [in] units
          const std::string StdName,     ///< [in] CF standard Name
          const std::any ValidMin,       ///< [in] min valid field value
          const std::any ValidMax,       ///< [in] max valid field value
          const std::any FillValue,      ///< [in] scalar for undefined entries
          const int NumDims,             ///< [in] number of dimensions
          const std::vector<std::string> Dimensions ///< dim names for each dim
   );

   /// Creates an empty metadata container for a field with input name
   static std::shared_ptr<MetaData>
   create(const std::string Name ///< [in] Name of field
   );

   /// Creates metadata for a scalar field with given name and list of
   /// (name,value) pairs
   static std::shared_ptr<MetaData>
   create(const std::string Name, ///< [in] Name of field
          const std::initializer_list<std::pair<std::string, std::any>>
              &MetaPairs /// < [in] List of (name,value) metadata pairs
   );

   /// Removes the metadata associated with a field of a given name
   static int destroy(const std::string Name /// Name of field to remove
   );

   /// Removes all defined metadata for all fields
   static void clear();

   /// Retrieves all metadata for a field with a given name
   static std::shared_ptr<MetaData>
   get(const std::string Name ///< [in] Name of field
   );

   /// Checks for the existence of a field with the given name
   static bool has(const std::string Name ///< [in] Name of field
   );

   /// Checks for the existence of a metadata entry with the given name
   bool hasEntry(const std::string Name ///< [in] Name of metadata entry
   );

   /// Adds a metadata entry with the (name,value) pair
   int addEntry(const std::string Name, ///< [in] Name of new metadata
                const std::any Value    ///< [in] Value of new metadata
   );

   /// Removes a metadata entry with the given name
   int removeEntry(const std::string Name ///< [in] Name of entry to remove
   );

   /// Returns the number of dimensions for the field
   int getNumDims();

   /// Returns a vector of dimension names associated with each dimension
   /// of an array field. Returns an error code.
   int getDimNames(
       std::vector<std::string> &Dimensions ///< [out] list of dimensions
   );

   /// Retrieves the value of the metadata associated with a given name
   /// This specific version of the overloaded interface coerces the value
   /// to an I4 type
   int getEntry(const std::string Name, ///< [in] Name of metadata to get
                I4 &Value               ///< [out] I4 Value of metadata
   );

   /// Retrieves the value of the metadata associated with a given name
   /// This specific version of the overloaded interface coerces the value
   /// to an I8 type
   int getEntry(const std::string Name, ///< [in] Name of metadata to get
                I8 &Value               ///< [out] I8 Value of metadata
   );

   /// Retrieves the value of the metadata associated with a given name
   /// This specific version of the overloaded interface coerces the value
   /// to an R4 type
   int getEntry(const std::string Name, ///< [in] Name of metadata to get
                R4 &Value               ///< [out]  R4 Value of metadata
   );

   /// Retrieves the value of the metadata associated with a given name
   /// This specific version of the overloaded interface coerces the value
   /// to an R8 type
   int getEntry(const std::string Name, ///< [in] Name of metadata to get
                R8 &Value               ///< [out] R8 Value of metadata
   );

   /// Retrieves the value of the metadata associated with a given name
   /// This specific version of the overloaded interface coerces the value
   /// to a bool type
   int getEntry(const std::string Name, ///< [in] Name of metadata to get
                bool &Value             ///< [out] bool Value of metadata
   );

   /// Retrieves the value of the metadata associated with a given name
   /// This specific version of the overloaded interface coerces the value
   /// to a string type
   int getEntry(const std::string Name, ///< [in] Name of metadata to get
                std::string &Value      ///< [out] string Value of metadata
   );

   /// returns a pointer to the metadata map of all entries
   std::map<std::string, std::any> *getAllEntries();

}; // end MetaData class

//----------------------------------------------------------------------------//
/// The MetaGroup class allows fields and their metadata to be grouped together
/// and referred to by a single name. This helps to reduce the length and
/// complexity of input lists and contents lists.
class MetaGroup {

 private:
   /// Name of group
   std::string GrpName;

   /// List of fields in group - only names are needed
   std::set<std::string> Fields;

   /// Store and maintain all defined groups in map container
   static std::map<std::string, std::shared_ptr<MetaGroup>> AllGroups;

 public:
   /// Creates an empty metadata group with a given name
   static std::shared_ptr<MetaGroup>
   create(const std::string Name ///< [in] Name of group to create
   );

   /// Removes a metadata group
   static int destroy(const std::string Name ///< [in] Name of group to destroy
   );

   /// Retrieves a pointer to a metadata group by name
   static std::shared_ptr<MetaGroup>
   get(const std::string Name ///< [in] Name of group to retrieve
   );

   /// Determines whether a group of a given name exists
   static bool has(const std::string Name ///< [in] Name of group
   );

   /// Determins whether a field of a given name exists in the group
   bool hasField(const std::string Name ///< [in] Name of field
   );

   /// Adds a field to the group
   int addField(const std::string Name ///< [in] Name of field to add
   );

   /// Retrieves a field's metadata from a group
   std::shared_ptr<MetaData>
   getField(const std::string Name ///< [in] Name of field to retrieve
   );

   /// Removes a field from a group
   int removeField(const std::string Name ///< [in] Name of field to remove
   );

   /// Returns a list of fields in a group
   std::set<std::string> getFieldList();

}; // end class MetaGroup

} // namespace OMEGA

#endif // OMEGA_METADATA_H
