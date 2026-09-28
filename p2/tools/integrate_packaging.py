"""Move Python extensions/resources into the single native vLLM namespace."""

from source_edit import ROOT, read, write, replace, replace_text

path = "p1_build.py"
replace(path, "package_names", '''
def resource_namespace(primary: str) -> str:
    return "vllm" if primary == "vllm" else primary + "_ascend"


def package_names(primary: str, addon: str) -> list[str]:
    packages = find_packages(str(ROOT), include=[primary, primary + ".*"])
    if addon != primary:
        packages += find_packages(str(ROOT / "ascend"), include=[addon, addon + ".*"])
    return packages
''')
source = read(path).replace('addon = primary + "_ascend"', 'addon = resource_namespace(primary)')
source = source.replace('for name in (primary, addon):', 'for name in dict.fromkeys((primary, addon)):')
source = source.replace('if addon == "vllm_ascend":', 'if primary == "vllm":')
source = source.replace('"vllm_ascend": [\n                "vllm_ascend_C*.so",', '"vllm": [\n                "_ascend_C*.so",')
source = source.replace('module = "vllm_ascend_C" if', 'module = "_ascend_C" if')
source = source.replace('"package_dir": {primary: primary, addon: f"ascend/{addon}"},', '"package_dir": ({primary: primary} if primary == addon else {primary: primary, addon: f"ascend/{addon}"}),')
source = source.replace('aclnn_root / "vllm_ascend/_cann_ops_custom"', 'aclnn_root / "vllm/_cann_ops_custom"')
source = source.replace('"""P1\'s Ascend-only', '"""P2 native Ascend')
write(path, source)
path = "p1_dev.py"
source = read(path).replace('addon = primary + "_ascend"', 'addon = builder.resource_namespace(primary)')
source = source.replace('for namespace in (primary, addon):', 'for namespace in dict.fromkeys((primary, addon)):')
# A wheel must not accidentally bundle the retained patch archive or plugin namespace.
source = source.replace('        meta = BytesParser().parsebytes(wheel.read(metas[0]))', '''        if primary == "vllm" and any(
            name.startswith(("vllm_ascend/", "ascend/legacy_patches/", "ascend/legacy_plugin/"))
            for name in names
        ):
            raise ValueError("P2 wheel contains a retired plugin namespace or patch archive")
        meta = BytesParser().parsebytes(wheel.read(metas[0]))''')
write(path, source)
for path in ("ascend/CMakeLists.txt", "ascend/csrc/camem_allocator.cpp"):
    p = ROOT / path
    p.write_text(p.read_text().replace("vllm_ascend_C", "_ascend_C"))
path = ROOT / "ascend/csrc/build_aclnn.sh"
path.write_text(path.read_text().replace("vllm_ascend/_cann_ops_custom", "vllm/_cann_ops_custom"))
path = ROOT / "MANIFEST.in"
path.write_text(path.read_text() + "\nprune ascend/legacy_patches\nprune ascend/legacy_plugin\nprune ascend/vllm_ascend\n")
for p in (ROOT / "vllm/_cann_ops_custom").rglob("__init__.py"):
    p.unlink()

path = "tests/standalone/test_p1_development.py"
source = read(path).replace('self.addon = self.primary + "_ascend"', 'self.addon = BUILD.resource_namespace(self.primary)')
source = source.replace('vllm_ascend_C', '_ascend_C').replace('"vllm_ascend/_cann_ops_custom', '"vllm/_cann_ops_custom')
write(path, source)
path = "ascend/tests/standalone/test_p1_resources.py"
write(path, read(path).replace('"vllm_ascend"', '"vllm"'))

print("Native extension name, CANN resources, wheel and strict-editable mappings updated")
