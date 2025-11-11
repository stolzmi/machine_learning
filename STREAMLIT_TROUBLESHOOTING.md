# Streamlit App Troubleshooting Guide

## Issue: Always Predicts "1"

### Problem
The app consistently predicts "1" regardless of what you draw.

### Root Cause
Image preprocessing was incorrect - the model wasn't seeing your drawing properly.

### Solution ✅ FIXED

The preprocessing function has been updated to:

1. **Extract RGB properly** (not just alpha channel)
2. **Find bounding box** of your drawing
3. **Crop and center** the digit
4. **Resize to 20×20** then center in 28×28 (matches MNIST format)
5. **Normalize** to [0, 1] range

### How to Use Fixed Version

```bash
# Just restart the app
streamlit run streamlit_mnist_app.py
```

### Verify It's Working

1. **Draw a digit** (any digit, not just 1)
2. **Click Predict**
3. **Check "XAI Visualization" tab**
4. **Look at "Preprocessed Image (28×28)"** - You should see your digit clearly

**What you should see**:
- White digit on black background
- Digit centered in 28×28 grid
- Clear, visible strokes

**If it still looks wrong**:
- Draw larger (use more of the canvas)
- Draw with thicker strokes
- Make sure digit is somewhat centered

## Common Issues and Solutions

### Issue 1: Preprocessed Image Looks Distorted

**Symptoms**:
- Digit is stretched or squashed
- Digit is tiny
- Digit is cut off

**Causes**:
- Drawing too small
- Drawing in corner
- Canvas too small

**Solutions**:
✅ **Draw larger** - Use at least 50% of canvas
✅ **Center drawing** - Don't draw in corners
✅ **Increase canvas size** - Try 350-400 pixels
✅ **Use thicker brush** - Try 25-30 pixels

### Issue 2: Preprocessed Image is Blank

**Symptoms**:
- See black square in "Model Input"
- Get warning about empty canvas

**Causes**:
- Strokes too thin
- Canvas cleared but still processing

**Solutions**:
✅ **Use thicker brush** (30-40 pixels)
✅ **Draw with multiple strokes**
✅ **Check you didn't accidentally clear**

### Issue 3: Low Confidence Predictions

**Symptoms**:
- Confidence below 70%
- Wrong predictions
- Top 3 predictions are close

**Causes**:
- Digit not clear enough
- Unusual drawing style
- Missing key features

**Solutions**:
✅ **Draw more clearly** - Bold, complete strokes
✅ **Match handwriting style** - Not printed font
✅ **Complete all strokes**:
   - For 0, 6, 8, 9: Close the loops
   - For 4: Clear intersection
   - For 7: Prominent top bar
   - For 8: Both loops visible

### Issue 4: Wrong Digit Predicted

**Symptoms**:
- Drew "4" but got "9"
- Drew "7" but got "1"
- Drew "3" but got "8"

**Debugging Steps**:

1. **Check preprocessed image**:
   - Does it look like what you drew?
   - Is it centered?
   - Are all strokes visible?

2. **Check XAI visualizations**:
   - Where is model focusing (GradCAM)?
   - What pixels matter (Saliency)?

3. **Check stroke features**:
   - Are expected features present?
   - Example: "4" should have high intersection score

4. **Check interpretation**:
   - What features did model find?
   - What's missing?

**Common Confusions**:
- **4 vs 9**: Make intersection clear in "4"
- **3 vs 8**: Make "3" open on left, "8" fully closed
- **1 vs 7**: Make "7" with clear top bar
- **5 vs 6**: Make bottom curve distinct

## Drawing Tips for Each Digit

### Digit 0
✅ **DO**: Draw circular/oval loop
✅ **DO**: Close the loop completely
❌ **DON'T**: Make too square

### Digit 1
✅ **DO**: Draw straight vertical line
✅ **DO**: Can add small hook at top
❌ **DON'T**: Make too thick (looks like 7)

### Digit 2
✅ **DO**: Clear top curve
✅ **DO**: Straight bottom base
❌ **DON'T**: Make curve too round (looks like 3)

### Digit 3
✅ **DO**: Two curves stacked
✅ **DO**: Open on left side
❌ **DON'T**: Close completely (looks like 8)

### Digit 4
✅ **DO**: **CLEAR INTERSECTION** (most important!)
✅ **DO**: Visible horizontal and vertical crossing
❌ **DON'T**: Make open at top (looks like 9)

### Digit 5
✅ **DO**: Horizontal top bar
✅ **DO**: Bottom curve
❌ **DON'T**: Make too curvy overall (looks like 6 or 3)

### Digit 6
✅ **DO**: Bottom loop clearly closed
✅ **DO**: Top stem/curve
❌ **DON'T**: Make too symmetric (looks like 8)

### Digit 7
✅ **DO**: **STRONG TOP BAR** (most important!)
✅ **DO**: Clear diagonal downstroke
❌ **DON'T**: Make vertical (looks like 1)

### Digit 8
✅ **DO**: Two clear loops (top and bottom)
✅ **DO**: Visible center connection
❌ **DON'T**: Make one loop much bigger (looks like 6 or 9)

### Digit 9
✅ **DO**: Clear top loop
✅ **DO**: Vertical stem down
❌ **DON'T**: Make top too open (looks like 4)

## Optimal Settings

### For Best Recognition

**Canvas Settings**:
- **Canvas Size**: 320-350 pixels
- **Brush Size**: 25-30 pixels

**Drawing Style**:
- Draw at **normal handwriting speed**
- Use **smooth, continuous strokes**
- **Center** your digit (roughly)
- Use **60-80%** of canvas area

### For Testing Different Digits

Try this progression:
1. Start with **1** (easiest - just vertical line)
2. Then **0** (simple circle)
3. Then **7** (test top bar recognition)
4. Then **4** (test intersection detection)
5. Then **8** (test multiple features)

This helps you understand what features the model looks for!

## Debug Checklist

When something goes wrong, check these in order:

- [ ] **Model loaded?** - See "Model loaded" message at startup
- [ ] **Canvas not empty?** - Drew something visible
- [ ] **Preprocessed image looks right?** - Check in XAI tab
- [ ] **Confidence reasonable?** - At least 50%
- [ ] **Right features detected?** - Check stroke features tab
- [ ] **Drawing matches MNIST style?** - Handwritten, not printed

## Still Having Issues?

### Try This Test

1. **Draw a simple "1"** (just vertical line)
   - Should get high confidence (>90%)
   - If not, check preprocessing

2. **Draw a "0"** (circle)
   - Should recognize as 0
   - Check if loop is closed

3. **Draw a "4"** (intersection important)
   - Check if intersection is detected
   - Look at stroke features

If all three work, preprocessing is correct!

### Advanced Debugging

**Check preprocessed image statistics**:

After drawing, look at the "Model Input" image in XAI tab:
- **Should see**: White digit on black background
- **Size**: 28×28 pixels
- **Position**: Roughly centered
- **Brightness**: Digit should be clearly visible

**Check model behavior**:
- Go to **Interpretation tab**
- Read **"Why This Prediction?"** section
- If confidence is low, it explains why
- Use this to adjust your drawing

## Performance Tips

### Make App Faster

1. **Close other tabs/programs** - Free up RAM
2. **Use smaller canvas** (280px) - Faster processing
3. **Draw decisively** - Don't redraw many times
4. **Clear between digits** - Fresh start each time

### Make Recognition Better

1. **Draw larger** - Use more canvas space
2. **Bold strokes** - Thicker brush, confident drawing
3. **Complete features** - Don't leave strokes unfinished
4. **Natural style** - Handwriting, not geometric shapes

## Summary

**Main fix**: Preprocessing now properly:
- Finds your drawing (bounding box)
- Crops and centers it
- Resizes to match MNIST format
- Preserves stroke thickness

**To verify working**: Look at "Preprocessed Image" in XAI tab - should clearly show your digit!

**If still having issues**: Try drawing larger, with thicker strokes, and more centered.

---

**The app should now work correctly for all digits 0-9!** ✅
