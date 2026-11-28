"""Regenerates Team3_MoshFlow.sln and MoshFlow.pyproj for Visual Studio 2019/2022.

Run `python tools/update_vs_project.py` after adding or removing source files.
New effects run without this (they are discovered from the folder), but they
only show up in Visual Studio's Solution Explorer after it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLUTION_NAME = "Team3_MoshFlow.sln"
PROJECT_NAME = "MoshFlow.pyproj"
PYTHON_PROJECT_TYPE = "{888888A0-9F3D-457C-B088-3A5042F75D52}"
PROJECT_GUID = "{6E0B6C5E-3F0A-4B7E-9C51-16A1D5F2B016}"
SOLUTION_GUID = "{B1C7A9D4-52E8-4F36-8A0D-16F4C3E7A916}"
SOURCE_DIRS = ("moshflow", "tests", "tools")
CONTENT_FILES = ("README.md", "CLAUDE.md", "docs/API.md", "requirements.txt", "pytest.ini", "run_windows.bat")


def source_files() -> list[Path]:
    files = [ROOT / "run.py"]
    for name in SOURCE_DIRS:
        files += [p for p in (ROOT / name).rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(files)


def folders(files: list[Path]) -> list[str]:
    found = set()
    for path in files:
        for parent in path.relative_to(ROOT).parents:
            if str(parent) != ".":
                found.add(_win(parent))
    return sorted(found)


def _win(path: Path) -> str:
    return str(path).replace("/", "\\")


def pyproj_text() -> str:
    files = source_files()
    compile_items = "\n".join(f'    <Compile Include="{_win(p.relative_to(ROOT))}" />' for p in files)
    folder_items = "\n".join(f'    <Folder Include="{f}\\" />'
                             for f in folders(files + [ROOT / name for name in CONTENT_FILES]))
    content_items = "\n".join(f'    <Content Include="{_win(Path(name))}" />' for name in CONTENT_FILES)
    return f"""<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003" ToolsVersion="4.0">
  <PropertyGroup>
    <Configuration Condition=" '$(Configuration)' == '' ">Debug</Configuration>
    <SchemaVersion>2.0</SchemaVersion>
    <ProjectGuid>{PROJECT_GUID}</ProjectGuid>
    <ProjectHome>.</ProjectHome>
    <StartupFile>run.py</StartupFile>
    <SearchPath>.</SearchPath>
    <WorkingDirectory>.</WorkingDirectory>
    <OutputPath>.</OutputPath>
    <Name>MoshFlow</Name>
    <RootNamespace>MoshFlow</RootNamespace>
    <IsWindowsApplication>False</IsWindowsApplication>
    <LaunchProvider>Standard Python launcher</LaunchProvider>
    <EnableNativeCodeDebugging>False</EnableNativeCodeDebugging>
    <TestFramework>Pytest</TestFramework>
  </PropertyGroup>
  <PropertyGroup Condition=" '$(Configuration)' == 'Debug' ">
    <DebugSymbols>true</DebugSymbols>
    <EnableUnmanagedDebugging>false</EnableUnmanagedDebugging>
  </PropertyGroup>
  <PropertyGroup Condition=" '$(Configuration)' == 'Release' ">
    <DebugSymbols>true</DebugSymbols>
    <EnableUnmanagedDebugging>false</EnableUnmanagedDebugging>
  </PropertyGroup>
  <ItemGroup>
{compile_items}
  </ItemGroup>
  <ItemGroup>
{folder_items}
  </ItemGroup>
  <ItemGroup>
{content_items}
  </ItemGroup>
  <Import Project="$(MSBuildExtensionsPath32)\\Microsoft\\VisualStudio\\v$(VisualStudioVersion)\\Python Tools\\Microsoft.PythonTools.targets" />
</Project>
"""


def sln_text() -> str:
    return f"""
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 16
VisualStudioVersion = 16.0.28729.10
MinimumVisualStudioVersion = 10.0.40219.1
Project("{PYTHON_PROJECT_TYPE}") = "MoshFlow", "{PROJECT_NAME}", "{PROJECT_GUID}"
EndProject
Global
\tGlobalSection(SolutionConfigurationPlatforms) = preSolution
\t\tDebug|Any CPU = Debug|Any CPU
\t\tRelease|Any CPU = Release|Any CPU
\tEndGlobalSection
\tGlobalSection(ProjectConfigurationPlatforms) = postSolution
\t\t{PROJECT_GUID}.Debug|Any CPU.ActiveCfg = Debug|Any CPU
\t\t{PROJECT_GUID}.Release|Any CPU.ActiveCfg = Release|Any CPU
\tEndGlobalSection
\tGlobalSection(SolutionProperties) = preSolution
\t\tHideSolutionNode = FALSE
\tEndGlobalSection
\tGlobalSection(ExtensibilityGlobals) = postSolution
\t\tSolutionGuid = {SOLUTION_GUID}
\tEndGlobalSection
EndGlobal
"""


def write(path: Path, text: str) -> None:
    # Visual Studio's own format: UTF-8 with BOM and CRLF line endings.
    path.write_text(text, encoding="utf-8-sig", newline="\r\n")
    print("wrote", path.name)


if __name__ == "__main__":
    write(ROOT / PROJECT_NAME, pyproj_text())
    write(ROOT / SOLUTION_NAME, sln_text())
