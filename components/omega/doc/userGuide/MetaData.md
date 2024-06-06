(omega-user-metadata)=

## Metadata

Metadata are additional data that are included in Omega input or output files
to create a self-describing file. Each field in a file will have attached
metadata that conforms to the netCDF
[Climate and Forecast metadata standards](https://cfconventions.org),
including fields like units, standard names, longer descriptions, min/max
range, fill values, etc. In addition, Omega allows global metadata that
describe either the code (eg version number) or the current experiment
being run (eg case or experiment name, simulation time). The MetaData
and MetaDim classes in Omega manage this information. There are no
user-configurable parameters for the classes themselves. Internally,
modules will define the metadata for all fields that they own and users
will specify those field names in the yaml configuration file when listing
contents of an [IOStream](#omega-user-iostreams). Other sections of the yaml
configuration will define code and simulation parameters that may be part of
global metadata. Finally, it is possible to define a metadata group (MetaGroup)
with a list of previously-defined fields as group members. This provides a
shortcut name for fields that are commonly grouped together and allows shorter
lists of contents when specifying contents of IOStreams.

The [Developer's Guide](#omega-dev-metadata) has more detail on the
class implementation and interfaces for creating and manipulating metadata.
