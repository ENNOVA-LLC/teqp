"""Regenerate model_schemas.cpp (binary-encoded JSON) and refresh the
dev/model_schemas.tar.xz tarball so the schema survives CMake reconfigures.

This is the manual equivalent of what dev/SchemaBuilder.ipynb does.
"""
from pathlib import Path
import tarfile

TEQP_ROOT = Path(r"c:\Users\cjsis\Documents\Github\ENNOVA\dev-technical\teqp")
SCHEMAS_JSON = TEQP_ROOT / "interface" / "CPP" / "model_schemas.json"
SCHEMAS_CPP = TEQP_ROOT / "interface" / "CPP" / "model_schemas.cpp"
TAR_XZ = TEQP_ROOT / "dev" / "model_schemas.tar.xz"

content = SCHEMAS_JSON.read_text(encoding="utf-8")
data = content.encode("utf-8")

# MSVC string-literal limit -> encode as a byte array. Format matches what
# model_schemas.cpp already looks like: 16 bytes per line, 0xNN comma-separated.
# A trailing 0x00 (null byte) is appended to match SchemaBuilder.ipynb's output;
# this is benign because nlohmann::json::parse stops at the closing brace.
hex_bytes = [f"0x{b:02x}" for b in data] + ["0x00"]
lines = []
for i in range(0, len(hex_bytes), 16):
    chunk = hex_bytes[i:i+16]
    lines.append(", ".join(chunk))
body = ",\n".join(lines)

cpp = (
    "// Due to limitations in MSVC, very long string literals are \n"
    "// not allowed. Thus the string must be re-encoded as binary. The \n"
    "// contents of the string are in the JSON file next to this file\n"
    "\n"
    "#include \"nlohmann/json.hpp\"\n"
    "#include <string>\n"
    "extern const auto model_schema_library = nlohmann::json::parse(std::string(\n"
    "{" + body + "}\n)); "
)
SCHEMAS_CPP.write_text(cpp, encoding="utf-8")
print(f"Wrote {SCHEMAS_CPP} ({len(data)} bytes of JSON content + trailing 0x00)")

# Re-pack the tarball so CMake reconfigure doesn't overwrite our changes.
# The tarball contains both .json and .cpp (see dev/SchemaBuilder.ipynb).
import os
cwd = os.getcwd()
try:
    os.chdir(SCHEMAS_JSON.parent)
    with tarfile.open(TAR_XZ, mode="w:xz") as tar:
        tar.add("model_schemas.json")
        tar.add("model_schemas.cpp")
finally:
    os.chdir(cwd)
print(f"Repacked {TAR_XZ}")
