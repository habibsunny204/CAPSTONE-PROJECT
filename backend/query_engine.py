"""Generic, schema-driven query engine: groupby_agg() and filtered_query() (Task A2).

Not yet implemented. Design: both functions take an explicit `con`, validate
caller-supplied dims/metrics/filter columns against the live schema, quote identifiers
safely, bind filter values as query parameters, and return (pandas.DataFrame,
elapsed_ms). No Superstore column names as literals in this file.
"""
