from .sql_depth import SQLDepth
from .sc_depth import SCDepth

method_map = {
    'sql-depth':SQLDepth,
    'sc-depth':SCDepth,
}