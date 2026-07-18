"""
Parse OLE2-format Excel files containing doctor time-segment annotations.

Excel format:
  - Old OLE2 binary format (.xls), despite the .xlsx extension
  - Sheet1 with header row followed by data rows
  - Columns: 医生id, 医生姓名, [视频内时间段1, 视频内时间段2, ...]
  - Time segment format: "M.SS-M.SS" (e.g., "2.06-2.31" = 2min06s to 2min31s)
"""

import xlrd
from config import VIDEO_FPS


def _parse_time(time_str: str) -> float:
    """
    Parse a time string in "M.SS" format to total seconds.

    Examples:
        "2.06" -> 126.0  (2 min 6 sec)
        "0.55" -> 55.0   (0 min 55 sec)
        "1.15" -> 75.0   (1 min 15 sec)
    """
    time_str = time_str.strip()
    minutes_str, seconds_str = time_str.split(".")
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    return minutes * 60.0 + seconds


def _parse_time_segment(segment_str: str) -> tuple[float, float]:
    """
    Parse "M.SS-M.SS" into (start_seconds, end_seconds).

    Example:
        "2.06-2.31" -> (126.0, 151.0)
    """
    segment_str = segment_str.strip()
    start_str, end_str = segment_str.split("-")
    return _parse_time(start_str), _parse_time(end_str)


def _time_to_frame(seconds: float, fps: int = VIDEO_FPS) -> int:
    """Convert seconds to frame index."""
    return int(seconds * fps)


def _normalize_doctor_id(val) -> str:
    """
    Normalize doctor ID to string.
    Handles both float (24502.0) and string ("24503") inputs.
    """
    if isinstance(val, float):
        return str(int(val))
    return str(val).strip()


def parse_excel(excel_path: str) -> list[dict]:
    """
    Parse a doctor info Excel file.

    Args:
        excel_path: Path to the OLE2-format .xlsx file

    Returns:
        List of dicts, each containing:
            - doctor_id: str, normalized doctor ID
            - name: str, doctor's Chinese name
            - segments: list of (start_frame, end_frame) tuples
    """
    workbook = xlrd.open_workbook(excel_path)
    sheet = workbook.sheet_by_name("Sheet1")

    doctors = []

    for row_idx in range(1, sheet.nrows):  # Skip header row
        doctor_id = _normalize_doctor_id(sheet.cell_value(row_idx, 0))
        doctor_name = str(sheet.cell_value(row_idx, 1)).strip()

        # Collect all non-empty time segment columns (starting from column 2)
        segments = []
        for col_idx in range(2, sheet.ncols):
            cell_val = str(sheet.cell_value(row_idx, col_idx)).strip()
            if cell_val and cell_val != "empty:":  # Skip empty cells
                try:
                    start_sec, end_sec = _parse_time_segment(cell_val)
                    start_frame = _time_to_frame(start_sec)
                    end_frame = _time_to_frame(end_sec)
                    segments.append((start_frame, end_frame))
                except (ValueError, IndexError) as e:
                    print(f"  [WARN] Failed to parse time segment '{cell_val}' "
                          f"for doctor {doctor_name} (row {row_idx + 1}): {e}")

        doctors.append({
            "doctor_id": doctor_id,
            "name": doctor_name,
            "segments": segments,
        })

    return doctors


if __name__ == "__main__":
    # Quick test
    from config import EXCEL_0427, EXCEL_0428
    import json

    for label, path in [("April 27", EXCEL_0427), ("April 28", EXCEL_0428)]:
        print(f"\n=== {label} ===")
        docs = parse_excel(path)
        for d in docs:
            print(f"  {d['doctor_id']} {d['name']}: {len(d['segments'])} segment(s)")
            for i, (sf, ef) in enumerate(d['segments']):
                print(f"    Segment {i + 1}: frames {sf}-{ef} ({sf / VIDEO_FPS:.1f}s - {ef / VIDEO_FPS:.1f}s)")
