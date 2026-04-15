# Fix: FFT Checkbox Remains Checked After Reset

## Problem Description

When clicking "Reset impostazioni", the "Calcola FFT" checkbox remained checked if it was previously enabled by the user, despite the reset button being designed to clear all settings.

## Root Cause Analysis

The issue was caused by a **race condition** in the widget initialization logic:

### Before Fix (line 719-727):
```python
if not fft_available:
    st.session_state["enable_fft"] = False  # ❌ Direct state write BEFORE widget creation

enable_fft = st.checkbox(
    "Calcola FFT",
    value=bool(st.session_state.get("enable_fft", False)),  # ❌ Reads from session state
    key="enable_fft",
    disabled=not fft_available,
    help=fft_help,
)
```

### Execution Flow:
1. User enables FFT → `st.session_state["enable_fft"] = True` (set by checkbox)
2. User clicks "Reset impostazioni" → `_reset_all_settings()` pops `"enable_fft"` from session state
3. Form recreates with new `_controls_nonce`
4. **Line 720 runs**: `st.session_state["enable_fft"] = False` only if `fft_available == False`
5. **Line 724 runs**: `value=bool(st.session_state.get("enable_fft", False))`
   - If FFT was available, line 720 didn't run, so the key is still absent
   - But Streamlit's internal form state REMEMBERS the old value from before the reset!
   - The `value=` parameter doesn't override Streamlit's form binding

### Why This Happens

Streamlit forms use **two-way data binding**:
- The `key=` parameter binds the widget to session state
- When a form is recreated (via nonce), Streamlit tries to preserve widget values **even if session state is cleared**
- The `value=` parameter is only used for **initial render**, not for form recreation

## Solution (Final Approach)

**Remove the `key=` parameter entirely** from the FFT checkbox:

### After Fix (line 748-754):
```python
# No key= parameter: widget state is local to the form, resets automatically
enable_fft = st.checkbox(
    "Calcola FFT",
    value=False,
    disabled=not fft_available,
    help=fft_help,
)
```

### Why This Works

1. **No `key=` binding**: Widget state is NOT stored in `st.session_state`
2. **Form nonce handles reset**: When `_controls_nonce` increments, the entire form recreates
3. **Widget recreates with `value=False`**: Fresh checkbox with unchecked state
4. **No memory leaks**: No keys accumulating in session state
5. **Simpler code**: No need to manage keys in `RESETTABLE_KEYS`

### Alternative Approaches Considered

❌ **Approach 1: Nonce-based key** (`key=f"enable_fft_{nonce}"`)
- **Problem**: Creates orphaned keys (`enable_fft_0`, `enable_fft_1`, etc.)
- **Memory leak**: Keys accumulate indefinitely in session state
- **Complexity**: Requires dynamic key generation

❌ **Approach 2: Session state read** (`value=st.session_state.get("enable_fft", False)`)
- **Problem**: Race condition with form binding
- **Streamlit caches**: Form state persists even when session state is cleared

✅ **Approach 3: No key (FINAL)** (`value=False`, no `key=`)
- **Clean**: Widget state is ephemeral, tied only to form lifecycle
- **Automatic reset**: Form nonce increment forces full widget recreation
- **Safe**: No session state pollution

## Verification

Test steps:
1. Load a CSV file with ≥128 rows (to make FFT available)
2. Expand "Advanced" section
3. Check "Calcola FFT"
4. Click "Applica / Plot" (FFT plots should appear)
5. Click "Reset impostazioni"
6. **Expected**: "Calcola FFT" should be **unchecked**
7. **Actual (before fix)**: Remained checked ❌
8. **Actual (after fix)**: Unchecked ✅

## Related Changes

This fix is part of the broader Issue #44 improvements:
- Added 3 missing state keys to reset (`_visual_report_prev_selection`, `_visual_report_last_default_x_label`, `_quality_file_sig`)
- Fixed FFT checkbox race condition
- Ensured all form widgets reset properly via `_controls_nonce` increment

## Files Modified

- `web_app.py` (lines 748-754)
  - Removed `if not fft_available: st.session_state["enable_fft"] = False`
  - Removed `key="enable_fft"` parameter from checkbox
  - Checkbox now uses only `value=False` without session state binding

- `web_app.py` (lines 30-61) - `RESETTABLE_KEYS`
  - Removed `"enable_fft"` from the set (no longer needed)

## Technical Notes

### Streamlit Form Behavior

When using `st.form()` with a dynamic key like `f"controls_{nonce}"`:
- Streamlit treats each nonce as a NEW form instance
- Widgets inside the form should **not** read from session state in their `value=` parameter
- Session state should only be written TO by widgets, not read FROM

### Best Practices for Resettable Forms

1. **Use form nonce**: Increment a nonce to force form recreation
   ```python
   with st.form(f"controls_{st.session_state.get('_controls_nonce', 0)}"):
   ```

2. **Don't read session state in widget values**:
   ```python
   # ❌ BAD
   enable = st.checkbox("Enable", value=st.session_state.get("enable", False), key="enable")

   # ✅ GOOD
   enable = st.checkbox("Enable", value=False, key="enable")
   ```

3. **Let widgets write to session state via key**:
   - The `key=` parameter automatically syncs widget state to `st.session_state`
   - Reading from it creates circular dependencies

4. **Clear keys on reset**:
   ```python
   for key in RESETTABLE_KEYS:
       st.session_state.pop(key, None)
   st.session_state["_controls_nonce"] = st.session_state.get("_controls_nonce", 0) + 1
   ```

## Status

- [x] Bug identified
- [x] Root cause analyzed
- [x] Multiple approaches evaluated
- [x] Final fix implemented (no-key approach)
- [x] Manual testing completed ✅
- [x] Verified working correctly
- [ ] Ready for commit (awaiting user confirmation)
