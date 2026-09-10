set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR="${1:-"$SCRIPT_DIR/../../artifacts"}"
PACKAGE_VERSION="${2:-1.4}"
WHEEL_TAG="${WHEEL_TAG:-cp312-cp312-linux_x86_64}"
if [[ ! "$WHEEL_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: WHEEL_TAG must contain only letters, digits, dots, underscores, or hyphens" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
BUILD_DIR=$(mktemp -d)
README_SOURCE="$SCRIPT_DIR/../README.md"
README_DESTINATION="$SCRIPT_DIR/deepstream_libraries/README.md"
if [[ ! -f "$README_SOURCE" ]]; then
  echo "error: DeepStream Libraries README not found at $README_SOURCE" >&2
  exit 1
fi
if [[ -e "$README_DESTINATION" ]]; then
  echo "error: package README already exists at $README_DESTINATION" >&2
  exit 1
fi

cleanup() {
  rm -rf -- "$BUILD_DIR"
  rm -f -- "$README_DESTINATION"
}
trap cleanup EXIT

cp "$README_SOURCE" "$README_DESTINATION"
cp -a "$SCRIPT_DIR/." "$BUILD_DIR/"

(
  cd "$BUILD_DIR"
  poetry version "$PACKAGE_VERSION"
  poetry build --output "$BUILD_DIR/dist"
)

SOURCE_WHEEL="$BUILD_DIR/dist/deepstream_libraries-${PACKAGE_VERSION}-py3-none-any.whl"

IFS='-' read -r PY_TAG ABI_TAG PLATFORM_TAG <<< "$WHEEL_TAG"
if [[ -z "$PY_TAG" || -z "$ABI_TAG" || -z "$PLATFORM_TAG" ]]; then
  echo "error: WHEEL_TAG must be in the form python-abi-platform, for example cp312-cp312-linux_x86_64" >&2
  exit 1
fi

python -m wheel tags \
  --python-tag "$PY_TAG" \
  --abi-tag "$ABI_TAG" \
  --platform-tag "$PLATFORM_TAG" \
  --remove \
  "$SOURCE_WHEEL"

RETAGGED_WHEEL="$BUILD_DIR/dist/deepstream_libraries-${PACKAGE_VERSION}-${WHEEL_TAG}.whl"
WHEEL_PATH="$OUTPUT_DIR/deepstream_libraries-${PACKAGE_VERSION}-${WHEEL_TAG}.whl"

mv -f "$RETAGGED_WHEEL" "$WHEEL_PATH"

printf 'Wheel written to: %s\n' "$WHEEL_PATH"
