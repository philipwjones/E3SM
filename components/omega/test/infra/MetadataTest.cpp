//===-- Test driver for OMEGA Metadata -------------------------*- C++ -*-===/
//
/// \file
/// \brief Test driver for OMEGA Metadata
///
/// This driver tests the Metadata capabilities for the OMEGA
/// model
//
//===-----------------------------------------------------------------------===/

#include <iostream>
#include <limits>

#include "DataTypes.h"
#include "Logging.h"
#include "MetaData.h"

using namespace OMEGA;

const I4 FILL_VALUE = 0;

void printResult(bool result, bool expected, std::string msg_pass,
                 std::string msg_fail) {

   if (result == expected) {
      std::cout << msg_pass << ": PASS" << std::endl;
   } else {
      std::cout << msg_fail << ": FAIL" << std::endl;
   }
}

void testMetaDim() {

   const std::string DimName{"MyDim"};
   const I4 DimValue{1};

   // test has()
   printResult(MetaDim::has(DimName), false, "'" + DimName + "' is not created",
               "'" + DimName + "' should not exist");

   // test create()
   auto Dim1 = MetaDim::create(DimName, DimValue);

   printResult(MetaDim::has(DimName), true, "'" + DimName + "' is created",
               "'" + DimName + "' should exist");

   // test get()
   auto DimNew = MetaDim::get(DimName);

   printResult(Dim1 == DimNew, true, "get() returns correct instance.",
               "get() returns incorrect instance.");

   // test getLength()
   I4 Length = Dim1->getLength();

   printResult(DimValue == Length, true, "getLength() returns correct length.",
               "getLength() returns incorrect length.");

   // create more dims and test ability to loop through
   std::vector<std::string> DimNames{"MyDim", "MyDim2", "MyDim3"};
   std::vector<I4> DimLengths{1, 2, 200};
   auto Dim2  = MetaDim::create(DimNames[1], DimLengths[1]);
   auto Dim3  = MetaDim::create(DimNames[2], DimLengths[2]);
   I4 NumDims = MetaDim::getNumDefinedDims();
   printResult(NumDims == 3, true, "Retrieved correct number of dims",
               "Retrieved incorrect number of dims");
   I4 DimCount = 0;
   I4 ErrCount = 0;
   for (auto It = MetaDim::begin(); It != MetaDim::end(); ++It) {
      std::string DimName = It->first;
      Length              = MetaDim::getDimLength(DimName);
      if (DimName != DimNames[DimCount])
         ++ErrCount;
      if (Length != DimLengths[DimCount])
         ++ErrCount;
      ++DimCount;
   }
   printResult(NumDims == DimCount, true, "MetaDim iterator correct count",
               "MetaDim iterator incorrect count");
   printResult(ErrCount == 0, true, "MetaDim iterator dereference correct",
               "MetaDim iterator dereference incorrect");

   // test destroy()
   MetaDim::destroy(DimName);

   printResult(MetaDim::has(DimName), false,
               "'" + DimName + "' is destroyed correctly",
               "'" + DimName + "' is not destroyed");

   // test clear
   MetaDim::clear();
   NumDims = MetaDim::getNumDefinedDims();
   printResult(NumDims == 0, true, "MetaDim clear removed all dims",
               "MetaDim clear did not remove all dims");
}

void testMetaData() {

   const std::string ArrayName{"MyArray"};
   const std::string DimName{"MyDim"};
   const I4 DimValue{1};
   int ret;

   // test has()
   printResult(MetaData::has(CodeMeta), false,
               "'" + CodeMeta + "' is not created",
               "'" + CodeMeta + "' should not exist");

   // test create() 1
   auto Data1 = MetaData::create(CodeMeta);

   printResult(MetaData::has(CodeMeta), true, "'" + CodeMeta + "' is created",
               "'" + CodeMeta + "' should exist");

   // test create() 2
   auto Data2 = MetaData::create(SimMeta, {std::make_pair("Meta1", 1),
                                           std::make_pair("Meta2", 2),
                                           std::make_pair("Meta3", 3)});

   printResult(MetaData::has(SimMeta), true, "'" + SimMeta + "' is created",
               "'" + SimMeta + "' should exist");

   std::map<std::string, std::any> *VarMeta = Data2->getAllEntries();
   std::string MetaName;
   int MetaVal, Count = 1;

   // use std::map functionality to loop through all metadata
   for (const auto &Pair : *VarMeta) {
      MetaName = Pair.first; // retrieve name part of meta pair
      MetaVal  = std::any_cast<int>(Pair.second); // retrieve value part

      printResult(MetaName == "Meta" + std::to_string(MetaVal), true,
                  "'" + SimMeta + "' has correct MetaName",
                  "'" + SimMeta + "' has wrong MetaName");

      printResult(MetaVal == Count, true,
                  "'" + SimMeta + "' has correct MetaVal",
                  "'" + SimMeta + "' has wrong MetaVal");

      // do whatever needed with metadata
      // metaVal can be cast into the appropriate type using
      //   std::any_cast<type>(metaVal)
      Count++;
   }

   // test create() 3 - Array metadata

   auto Dim1 = MetaDim::create(DimName, DimValue);

   std::vector<std::string> Dimensions;
   Dimensions.push_back(DimName);

   auto Data3 =
       MetaData::create(ArrayName,
                        "Description", /// long Name or description
                        "Units",       /// units
                        "StdName",     /// CF standard Name
                        std::numeric_limits<int>::min(), /// min valid value
                        std::numeric_limits<int>::max(), /// max valid value
                        FILL_VALUE, /// scalar used for undefined entries
                        1,          /// number of dimensions
                        Dimensions  /// dimension names
       );

   printResult(MetaData::has(ArrayName), true, "'" + ArrayName + "' is created",
               "'" + ArrayName + "' should exist");

   // test get()
   auto Data4 = MetaData::get(ArrayName);

   printResult(Data3 == Data4, true, "get() returns correct instance.",
               "get() returns incorrect instance.");

   // Get the number of dimensions for the field
   I4 NumDims = Data4->getNumDims();

   printResult(NumDims == 1, true,
               "MetaData.getNumDims() returns correct number of dimensions.",
               "MetaData.getNumDims() returns incorrect number of dimensions.");

   // Get the names of all dimensions
   std::vector<std::string> DimNames;
   ret = Data4->getDimNames(DimNames);
   printResult(ret == 0, true, "MetaData getDimNames successfully returned.",
               "MetaData getDimNames returned error.");

   Count = 0;
   for (int I = 0; I < NumDims; ++I) {
      if (DimNames[I] != Dimensions[I]) {
         LOG_ERROR("Retrived DimName {} does not match Dimension {} Index {}",
                   DimNames[I], Dimensions[I], I);
         ++Count;
      }
   }
   printResult(Count == 0, true, "MetaData retrieved correct dimension names.",
               "MetaData retrieved incorrect dimension names.");

   // test hasEntry()
   printResult(Data4->hasEntry("FillValue"), true,
               "'" + ArrayName + "' has a fill value.",
               "'" + ArrayName + "' does not have a fill value");

   // test getEntry()
   I4 FillValue;
   ret = Data3->getEntry("FillValue", FillValue);

   printResult(ret == 0, true, "getEntry() returns zero",
               "getEntry() returns non-zero");

   printResult(FILL_VALUE == FillValue, true,
               "'" + ArrayName + "' has a correct fill value",
               "'" + ArrayName + "' has an incorrect fill value");

   // test addEntry()
   const R8 NewValue = 2.0;
   ret               = Data3->addEntry("NewMeta", NewValue);

   printResult(ret == 0, true, "addEntry() returns zero",
               "addEntry() returns non-zero");

   R8 R8Value;
   ret = Data3->getEntry("NewMeta", R8Value);

   printResult(NewValue == R8Value, true, "getEntry() returns correct value",
               "getEntry() returns incorrect value");

   // test removeEntry()
   ret = Data3->removeEntry("NewMeta");

   printResult(ret == 0, true, "removeEntry() returns zero",
               "removeEntry() returns non-zero");

   printResult(Data3->hasEntry("NewMeta"), false,
               "'NewMeta' is removed correctly", "'NewMeta' is not removed");

   // test destroy()
   ret = MetaData::destroy(SimMeta);

   printResult(ret == 0, true, "destroy() returns zero",
               "destroy() returns non-zero");

   printResult(MetaData::has(SimMeta), false,
               "'" + SimMeta + "' is correclty removed",
               "'" + SimMeta + "' is not removed.");

   // test clear()
   MetaData::clear();

   printResult(MetaData::has(CodeMeta), false,
               "MetaData clear() - CodeMeta correclty removed",
               "MetaData clear() - CodeMeta is not removed.");
}

void testMetaGroup() {

   const std::string GroupName{"MyGroup"};
   const std::vector<std::string> FieldNames{"MyField1", "MyField2"};
   const std::string DimName{"MyDim"};
   const I4 DimValue{1};
   int ret;

   // test has()
   printResult(MetaGroup::has(GroupName), false,
               "'" + GroupName + "' is not created",
               "'" + GroupName + "' should not exist");

   // test create()
   auto Group1 = MetaGroup::create(GroupName);

   printResult(MetaGroup::has(GroupName), true,
               "'" + GroupName + "' is created",
               "'" + GroupName + "' should exist");

   // test get()
   auto Group2 = MetaGroup::get(GroupName);

   printResult(Group1 == Group2, true, "get() returns correct instance.",
               "get() returns incorrect instance.");

   // test hasField()
   printResult(Group1->hasField(FieldNames[0]), false,
               "'" + FieldNames[0] + "' is not in a group",
               "'" + FieldNames[0] + "' is in a group");

   // test addField()
   auto Data1 = MetaData::create(FieldNames[0]);
   auto Data2 = MetaData::create(FieldNames[1]);

   ret = Group1->addField(FieldNames[0]);

   printResult(ret == 0, true, "addField() returns zero.",
               "addField() returns non-zero.");

   printResult(Group1->hasField(FieldNames[0]), true,
               "'" + FieldNames[0] + "' is in a group",
               "'" + FieldNames[0] + "' is not in a group");

   // add a second field to test some retrievals
   ret = Group1->addField(FieldNames[1]);

   std::set<std::string> Fields = Group1->getFieldList();

   // use std::set functionality to loop through all fields
   int FieldCount{0};
   for (auto IField = Fields.begin(); IField != Fields.end(); ++IField) {
      printResult(*IField == FieldNames[FieldCount], true,
                  "Correct FieldName is returned",
                  "Incorrect FieldName is returned");
      ++FieldCount;
   }

   // test getField()
   auto Data3 = Group1->getField(FieldNames[0]);

   printResult(Data1 == Data3, true, "getField() returns correct instance.",
               "getField() returns incorrect instance.");

   // test removeField()
   ret = Group1->removeField(FieldNames[0]);

   printResult(ret == 0, true, "removeField() returns zero.",
               "removeField() returns non-zero.");

   printResult(Group1->hasField(FieldNames[0]), false,
               "'" + FieldNames[0] + "' is not in a group",
               "'" + FieldNames[0] + "' is in a group");

   // test destroy()
   MetaGroup::destroy(GroupName);

   printResult(MetaGroup::has(GroupName), false,
               "'" + GroupName + "' is destroyed correctly",
               "'" + GroupName + "' is not destroyed");
}

std::vector<std::shared_ptr<MetaDim>> initMetaDim(const std::string DimName,
                                                  const I4 DimValue) {

   std::vector<std::shared_ptr<MetaDim>> Dimensions;
   Dimensions.push_back(MetaDim::create(DimName, DimValue));

   return Dimensions;
}

void initMetaData(const std::string FieldName,
                  const std::vector<std::string> Dimensions) {

   auto Data =
       MetaData::create(FieldName,
                        "Description", /// long Name or description
                        "Units",       /// units
                        "StdName",     /// CF standard Name
                        std::numeric_limits<int>::min(), /// min valid value
                        std::numeric_limits<int>::max(), /// max valid value
                        FILL_VALUE, /// scalar used for undefined entries
                        1,          /// number of dimensions
                        Dimensions  /// dim names
       );
}

void initMetaGroup(const std::string GroupName) {

   auto Group1 = MetaGroup::create(GroupName);
}

void testMetaInit() {

   const std::string GroupName{"MyInitGroup"};
   const std::string FieldName{"MyInitField"};
   const std::string DimName{"MyInitDim"};
   std::vector<std::string> DimNames;
   const I4 DimValue{1};
   int ret;

   auto dimensions = initMetaDim(DimName, DimValue);
   DimNames.push_back(DimName);

   initMetaData(FieldName, DimNames);

   initMetaGroup(GroupName);

   printResult(MetaGroup::has(GroupName), true,
               "'" + GroupName + "' is created",
               "'" + GroupName + "' should exist");

   // test get()
   auto Group1 = MetaGroup::get(GroupName);

   // test hasField()
   printResult(Group1->hasField(FieldName), false,
               "'" + FieldName + "' is not in a group",
               "'" + FieldName + "' is in a group");

   ret = Group1->addField(FieldName);

   printResult(ret == 0, true, "addField() returns zero.",
               "addField() returns non-zero.");

   printResult(Group1->hasField(FieldName), true,
               "'" + FieldName + "' is in a group",
               "'" + FieldName + "' is not in a group");

   // test getField()
   auto Data1 = MetaData::get(FieldName);
   auto Data2 = Group1->getField(FieldName);

   printResult(Data1 == Data2, true, "getField() returns correct instance.",
               "getField() returns incorrect instance.");

   // test removeField()
   ret = Group1->removeField(FieldName);

   printResult(ret == 0, true, "removeField() returns zero.",
               "removeField() returns non-zero.");

   printResult(Group1->hasField(FieldName), false,
               "'" + FieldName + "' is not in a group",
               "'" + FieldName + "' is in a group");

   // test MetaGroup::destroy()
   MetaGroup::destroy(GroupName);

   printResult(MetaGroup::has(GroupName), false,
               "'" + GroupName + "' is destroyed correctly",
               "'" + GroupName + "' is not destroyed");

   // test MetaData::destroy()
   MetaData::destroy(FieldName);

   printResult(MetaData::has(FieldName), false,
               "'" + FieldName + "' is destroyed correctly",
               "'" + FieldName + "' is not destroyed");

   // test MetaDim::destroy()
   MetaDim::destroy(DimName);

   printResult(MetaDim::has(DimName), false,
               "'" + DimName + "' is destroyed correctly",
               "'" + DimName + "' is not destroyed");
}

int main(int argc, char **argv) {

   int RetVal = 0;

   try {

      testMetaDim();

      testMetaData();

      testMetaGroup();

      testMetaInit();

   } catch (const std::exception &Ex) {
      std::cout << Ex.what() << ": FAIL" << std::endl;
      RetVal -= -1;

   } catch (...) {
      std::cout << "Unknown: FAIL" << std::endl;
      RetVal -= -1;
   }

   return RetVal;
}
