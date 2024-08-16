#include "LayerThicknessAuxVars.h"
#include "DataTypes.h"
#include "Field.h"

#include <limits>

namespace OMEGA {

LayerThicknessAuxVars::LayerThicknessAuxVars(const std::string &AuxStateSuffix,
                                             const HorzMesh *Mesh,
                                             int NVertLevels)
    : FluxLayerThickEdge("FluxLayerThickEdge" + AuxStateSuffix,
                         Mesh->NEdgesSize, NVertLevels),
      MeanLayerThickEdge("MeanLayerThickEdge" + AuxStateSuffix,
                         Mesh->NEdgesSize, NVertLevels),
      CellsOnEdge(Mesh->CellsOnEdge) {}

void LayerThicknessAuxVars::registerFields(const std::string &AuxGroupName,
                                           const std::string &MeshName) const {

   int Err = 0; // Error flag for some calls

   // Create/define fields
   const Real FillValue = -9.99e30;
   int NDims            = 2;
   std::vector<std::string> DimNames(NDims);
   std::string DimSuffix;
   if (MeshName == "Default") {
      DimSuffix = "";
   } else {
      DimSuffix = MeshName;
   }

   DimNames[0] = "NEdges" + DimSuffix;
   DimNames[1] = "NVertLevels";

   // Flux layer thickness on edges
   auto FluxLayerThickEdgeField = Field::create(
       FluxLayerThickEdge.label(), // field name
       "layer thickness used for fluxes through edges. May be centered, "
       "upwinded, or a combination of the two.", // long Name or description
       "m",                                      // units
       "",                                       // CF standard Name
       0,                                        // min valid value
       std::numeric_limits<Real>::max(),         // max valid value
       FillValue,                                // scalar for undefined entries
       NDims,                                    // number of dimensions
       DimNames                                  // dimension names
   );

   // Mean layer thickness on edges
   auto MeanLayerThickEdgeField = Field::create(
       MeanLayerThickEdge.label(),                           // field name
       "layer thickness averaged from cell center to edges", // long Name or
                                                             // description
       "m",                                                  // units
       "",                                                   // CF standard Name
       0,                                                    // min valid value
       std::numeric_limits<Real>::max(),                     // max valid value
       FillValue, // scalar used for undefined entries
       NDims,     // number of dimensions
       DimNames   // dimension names
   );

   // Add fields to Aux field group
   Err = FieldGroup::addFieldToGroup(FluxLayerThickEdge.label(), AuxGroupName);
   if (Err != 0)
      LOG_ERROR("Error adding field {} to group {}", FluxLayerThickEdge.label(),
                AuxGroupName);

   Err = FieldGroup::addFieldToGroup(MeanLayerThickEdge.label(), AuxGroupName);
   if (Err != 0)
      LOG_ERROR("Error adding field {} to group {}", MeanLayerThickEdge.label(),
                AuxGroupName);

   // Attach field data
   Err = FluxLayerThickEdgeField->attachData<Array2DReal>(FluxLayerThickEdge);
   if (Err != 0)
      LOG_ERROR("Error attaching data to field {}", FluxLayerThickEdge.label());

   Err = MeanLayerThickEdgeField->attachData<Array2DReal>(MeanLayerThickEdge);
   if (Err != 0)
      LOG_ERROR("Error attaching data to field {}", MeanLayerThickEdge.label());
}

void LayerThicknessAuxVars::unregisterFields() const {

   int Err = 0;

   Err = Field::destroy(FluxLayerThickEdge.label());
   if (Err != 0)
      LOG_ERROR("Error destroying field {}", FluxLayerThickEdge.label());

   Err = Field::destroy(MeanLayerThickEdge.label());
   if (Err != 0)
      LOG_ERROR("Error destroying field {}", MeanLayerThickEdge.label());
}

} // namespace OMEGA
