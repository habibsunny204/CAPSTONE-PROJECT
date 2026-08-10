"""SQL validation and safe execution sandbox (Task B2).

Not yet implemented. Design: sqlglot-based validation enforcing single-SELECT-only,
rejecting ATTACH/DETACH/PRAGMA/INSTALL/LOAD/COPY/EXPORT, any DDL/DML, multiple
statements, and out-of-scope file reads, then executes against a read-only DuckDB
connection. See PROJECT_SPEC.md Section 7 for the full rejection list.
"""
