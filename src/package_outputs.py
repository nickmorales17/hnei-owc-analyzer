"""Reproducibility metadata, file manifest, and ZIP bundle creation."""

from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
import zipfile

import pandas as pd
import yaml


def sha256_file(path:str|Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def git_commit()->str:
    try: return subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError): return "unavailable"


def build_reproducibility_metadata(input_path:str|Path,config_path:str|Path,config:dict[str,Any],output_dir:Path,worksheet:str|None,total_records:int,available_channels:list[str],invocation:list[str]|None=None,time_metadata:dict[str,Any]|None=None)->dict[str,Any]:
    packages={name:importlib.metadata.version(name) for name in ["pandas","numpy","scipy","matplotlib","openpyxl","PyYAML"]}
    expected=["Pressure_1","Pressure_2","Pressure_3","Pressure_4","Torque","Encoder","Gen_V"]
    vfd=config.get("vfd_command",{})
    result={"input_filename":Path(input_path).name,"input_sha256":sha256_file(input_path),"configuration_filename":str(config_path),"configuration_sha256":sha256_file(config_path),"configuration_profile":config.get("profile_name","default"),"analysis_timestamp":datetime.now(timezone.utc).isoformat(),"python_version":platform.python_version(),"dependency_versions":packages,"git_commit":git_commit(),"command_line_invocation":" ".join(invocation or sys.argv),"worksheet_analyzed":worksheet,"total_records":int(total_records),"available_channels":available_channels,"missing_expected_channels":[c for c in expected if c not in available_channels],"vfd_scaling_status":"verified_configured" if vfd.get("enabled") else "unavailable_unverified_scaling","output_directory":str(output_dir.resolve())}
    if time_metadata:
        result.update({key:time_metadata.get(key) for key in ["time_source","time_reconstructed_flag","assumed_sample_interval_s","duration_s","absolute_timestamp_range","sampling_jitter_available","record_gap_count","timing_dependency_note"]})
    return result


def save_reproducibility(output_dir:Path,metadata:dict[str,Any],config:dict[str,Any])->tuple[Path,Path]:
    metadata_path=output_dir/"reproducibility_metadata.json"; metadata_path.write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
    config_path=output_dir/"configuration_used.yaml"; config_path.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    return metadata_path,config_path


def _description(relative:Path)->str:
    text=str(relative)
    if text.startswith("cleaned/"): return "Cleaned/annotated data export; raw measured fields preserved."
    if text.startswith("tables/"): return "Stage 1–6 analysis table."
    if text.startswith("graphs/"): return "Diagnostic or final engineering graph."
    if text.startswith("report/"): return "Engineering report deliverable."
    if relative.name=="analysis_summary.xlsx": return "Formatted Microsoft Excel engineering summary."
    if relative.name=="reproducibility_metadata.json": return "Machine-readable reproducibility metadata."
    if relative.name=="configuration_used.yaml": return "Resolved configuration used for this analysis."
    if relative.name=="run_log.txt": return "Analysis execution log."
    if relative.name=="file_manifest.txt": return "Generated-file inventory."
    return "Generated analysis artifact."


def create_file_manifest(output_dir:Path)->tuple[Path,pd.DataFrame]:
    rows=[]
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name!="file_manifest.txt"):
        rel=path.relative_to(output_dir); rows.append({"Relative_Path":str(rel),"File_Type":path.suffix.lower().lstrip(".") or "text","File_Size_Bytes":path.stat().st_size,"Description":_description(rel)})
    rows.append({"Relative_Path":"file_manifest.txt","File_Type":"txt","File_Size_Bytes":0,"Description":"Generated-file inventory (self-entry size omitted to avoid recursion)."})
    frame=pd.DataFrame(rows); manifest=output_dir/"file_manifest.txt"
    lines=["Relative path\tFile type\tFile size (bytes)\tDescription",*[f"{r.Relative_Path}\t{r.File_Type}\t{r.File_Size_Bytes}\t{r.Description}" for r in frame.itertuples(index=False)]]; manifest.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return manifest,frame


def create_analysis_bundle(output_dir:Path)->Path:
    bundle=output_dir/"analysis_bundle.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p!=bundle): archive.write(path,path.relative_to(output_dir))
    with zipfile.ZipFile(bundle) as archive:
        bad=archive.testzip(); assert bad is None
        names=set(archive.namelist()); assert "analysis_summary.xlsx" in names and "file_manifest.txt" in names and "reproducibility_metadata.json" in names
        assert not any(name.endswith((".xlsx",".xls")) and name!="analysis_summary.xlsx" for name in names)
    return bundle
