Please use
find . -type f \( -name "*.exe" -o -name "*.dll" \) -exec upx -d {} +
before using this script