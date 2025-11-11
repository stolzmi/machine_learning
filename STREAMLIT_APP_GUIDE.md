# 🎨 Interactive MNIST XAI Streamlit App

## Draw Digits and See Real-Time XAI Analysis!

A beautiful, interactive web app where you can draw handwritten digits and instantly see:
- ✅ Model prediction
- ✅ GradCAM visualization
- ✅ Saliency maps
- ✅ Regional importance analysis
- ✅ Stroke feature analysis
- ✅ Shape interpretation
- ✅ Confidence scores

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit-drawable-canvas
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 2. Train Model (If Not Done Already)

```bash
python train_mnist.py
```

This creates `mnist_model.pkl` which the app needs.

### 3. Launch the App

```bash
streamlit run streamlit_mnist_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 App Features

### Drawing Canvas
- **Adjustable canvas size**: 200-400 pixels
- **Adjustable brush size**: 10-50 pixels
- **Black background, white pen** (matches MNIST style)
- **Clear button**: Start over
- **Predict button**: Analyze your drawing

### Real-Time Analysis Tabs

#### 1. 🔍 XAI Visualization
Shows three views side-by-side:
- Your original drawing (28×28 processed)
- GradCAM heatmap (spatial importance)
- Saliency map (pixel importance)

#### 2. 📍 Regional Analysis
Interactive chart showing importance of:
- Top region
- Bottom region
- Left region
- Right region
- Center region

**Plus**: Top 3 most important regions listed

#### 3. ✏️ Stroke Features
Bar chart showing importance of:
- Horizontal strokes
- Vertical strokes
- Curves/loops
- Intersections

**Plus**: Dominant feature highlighted with expected patterns

#### 4. 📊 All Probabilities
Horizontal bar chart showing:
- Probability for each digit 0-9
- Predicted digit highlighted in dark green
- Top 3 predictions listed below

#### 5. 💡 Interpretation
Comprehensive text explanation:
- What features were found
- Why this classification was made
- Spatial focus areas
- Shape features detected
- Confidence level explanation

### Additional Features

**Confidence Indicators**:
- 🟢 High confidence (>90%): Clear, distinctive features
- 🔵 Good confidence (70-90%): Key features identified
- 🟡 Lower confidence (<70%): Ambiguous or unclear

**Download Report**:
- Save full analysis as text file
- Includes predictions, confidence, and interpretation

## 📖 How to Use

### Step-by-Step Guide

1. **Launch the app**:
   ```bash
   streamlit run streamlit_mnist_app.py
   ```

2. **Adjust settings** (sidebar):
   - Set canvas size (bigger = easier to draw)
   - Set brush size (thicker = clearer digits)

3. **Draw a digit**:
   - Use your mouse or trackpad
   - Draw in the center of the canvas
   - Make strokes clear and bold
   - Try to match handwriting style

4. **Click "Predict"**:
   - App processes your drawing
   - Resizes to 28×28 (MNIST format)
   - Runs through model
   - Generates all analyses

5. **Explore results**:
   - Check prediction at top (big number)
   - Switch between tabs to see different analyses
   - Read interpretation to understand "why"

6. **Try again**:
   - Click "Clear" to start over
   - Draw a different digit
   - Compare how different digits are analyzed

## 🎯 Tips for Best Results

### Drawing Tips

✅ **DO**:
- Draw larger digits (use more canvas space)
- Center your drawing
- Make clear, bold strokes
- Draw like you're handwriting (natural style)
- Complete all strokes (closed loops for 0, 6, 8, 9)

❌ **DON'T**:
- Draw too small (hard to recognize)
- Draw in corners (gets cut off)
- Use thin lines (may disappear when resized)
- Draw printed-style letters (model trained on handwriting)

### Digit-Specific Tips

**Digit "1"**:
- Draw a clear vertical line
- Can add a small hook at top

**Digit "4"**:
- Make the intersection clear
- Ensure horizontal and vertical cross

**Digit "7"**:
- Strong horizontal bar at top
- Clear diagonal stroke

**Digit "8"**:
- Make both loops visible
- Show clear center intersection

**Digit "9"**:
- Complete top loop
- Clear vertical stem at bottom

## 🔍 Understanding the Analysis

### GradCAM Interpretation

**Red/Yellow regions** = Model focuses here
**Blue/Purple regions** = Model ignores these

**Example - Digit "4"**:
```
Expected pattern:
- Red at intersection point
- Yellow on crossing strokes
- Blue in empty corners
```

### Saliency Map Interpretation

**Bright pixels** = Very important
**Dark pixels** = Not important

**Example - Digit "8"**:
```
Expected pattern:
- Bright on all curves
- Very bright at center
- Dark in empty spaces
```

### Regional Analysis

Shows which parts of the 28×28 image matter most.

**Example readings**:
- High "top" score → Digit has important top features (7, 5)
- High "center" score → Digit has central features (4, 8)
- High "bottom" score → Digit has bottom features (2, 6)

### Stroke Features

Shows which geometric patterns were found.

**Interpretation**:
- High horizontal → Bars/lines (7, 5, 2)
- High vertical → Vertical lines (1, 4)
- High curves → Loops (0, 6, 8, 9)
- High intersections → Crossing points (4, 8)

## 💡 Example Analysis Walkthrough

### Example: Drawing a "4"

1. **Draw the digit**:
   - Horizontal line
   - Vertical line crossing it
   - Make intersection clear

2. **Click Predict**:
   ```
   Predicted Digit: 4
   Confidence: 92.5%
   ```

3. **Check XAI Visualization**:
   - GradCAM: Red at intersection point
   - Saliency: Bright on both strokes

4. **Check Regional Analysis**:
   ```
   Top regions:
   1. center: 0.823
   2. top-left: 0.645
   3. right: 0.534
   ```

5. **Check Stroke Features**:
   ```
   Intersections: 0.845 (dominant!)
   Vertical: 0.623
   Horizontal: 0.501
   Curves: 0.234
   ```

6. **Read Interpretation**:
   ```
   "Model found clear intersection point
   (key feature of '4') with both horizontal
   and vertical strokes present."
   ```

### Example: Ambiguous Drawing

1. **Draw poorly** (e.g., 4 that looks like 9)

2. **Click Predict**:
   ```
   Predicted Digit: 9
   Confidence: 58.3%
   ```

3. **Check probabilities**:
   ```
   Top 3:
   1. Digit 9: 58.3%
   2. Digit 4: 35.1%
   3. Digit 7: 4.2%
   ```

4. **Read Interpretation**:
   ```
   ⚠️ Lower Confidence
   Model is uncertain. Missing clear
   intersection that defines '4'.
   Top portion resembles '9'.
   ```

5. **Action**: Redraw with clearer intersection!

## 🎨 Use Cases

### 1. Educational Demonstrations
- Show students how CNNs work
- Visualize decision-making process
- Explain XAI concepts interactively

### 2. Model Testing
- Test edge cases (ambiguous digits)
- Find what confuses the model
- Validate model behavior

### 3. Data Collection Insights
- See what features model learned
- Identify biases or gaps
- Guide data augmentation strategy

### 4. Presentations
- Live demo for stakeholders
- Interactive explanation of AI
- Build trust through transparency

### 5. Debugging
- Test specific digit styles
- Understand misclassifications
- Improve model based on insights

## 🔧 Customization

### Adjust Canvas Settings

In the sidebar, you can change:

**Canvas Size** (200-400):
- Smaller: Faster, but harder to draw
- Larger: Easier to draw, more detail

**Brush Size** (10-50):
- Thinner: More precise, but may disappear
- Thicker: Clearer, better for recognition

### Recommended Settings

**For precise drawing**:
- Canvas: 350 pixels
- Brush: 15-20 pixels

**For quick testing**:
- Canvas: 280 pixels
- Brush: 20-25 pixels

**For bold strokes**:
- Canvas: 400 pixels
- Brush: 30-40 pixels

## 📊 Technical Details

### Image Processing Pipeline

1. **Canvas Input**: 280×280 (or custom size) RGBA
2. **Extract Alpha**: Get drawing from alpha channel
3. **Invert Colors**: Black on white (MNIST format)
4. **Resize**: Down to 28×28 using Lanczos interpolation
5. **Normalize**: Scale to [0, 1]
6. **Add Channel**: Shape becomes (28, 28, 1)

### Analysis Pipeline

1. **Prediction**: Forward pass through CNN
2. **GradCAM**: Compute on 'conv3' layer
3. **Saliency**: Compute gradients w.r.t. input
4. **Regions**: Divide into 9 regions, analyze each
5. **Strokes**: Analyze horizontal, vertical, curves, intersections
6. **Interpretation**: Generate human-readable explanation

### Performance

- **Prediction Time**: <1 second
- **Full Analysis**: 2-3 seconds
- **Visualization Generation**: <1 second

## 🐛 Troubleshooting

### Issue: Model Not Found
```
Error: Model file 'mnist_model.pkl' not found!
```
**Solution**: Train the model first
```bash
python train_mnist.py
```

### Issue: Canvas Not Responding
**Solution**:
- Refresh the browser
- Check console for errors
- Ensure streamlit-drawable-canvas is installed

### Issue: Poor Recognition
**Possible causes**:
- Drawing too small
- Drawing off-center
- Lines too thin
- Style too different from training data

**Solutions**:
- Draw larger
- Center your digit
- Use thicker brush
- Draw in natural handwriting style

### Issue: Wrong Predictions
**Debug steps**:
1. Check the processed 28×28 image (shown in XAI tab)
2. See if it looks like the digit you drew
3. If not, adjust canvas/brush size
4. If yes, model may be confused - check interpretation tab

## 🌟 Advanced Features

### Understanding Confidence

The app shows three confidence levels:

**High (>90%)**:
- ✅ Clear features found
- Strong match to training data
- Very likely correct

**Good (70-90%)**:
- ℹ️ Key features identified
- Some ambiguity present
- Probably correct

**Low (<70%)**:
- ⚠️ Uncertain
- Missing features or ambiguous
- Check alternatives in probability tab

### Multi-Digit Comparison

Try drawing multiple digits and compare:

1. Draw "4" → Save analysis
2. Clear and draw "9" → Compare
3. Notice differences in:
   - Regional focus
   - Stroke features
   - Shape patterns

### Testing Edge Cases

**Ambiguous digits**:
- Draw 4 that looks like 9
- Draw 3 that looks like 8
- Draw 1 that looks like 7

See how model handles confusion!

## 📝 Report Download

Click "Save Analysis Report" to download a text file containing:

```
MNIST XAI Analysis Report
========================

Predicted Digit: 4
Confidence: 92.50%

Top 3 Predictions:
1. Digit 4: 92.50%
2. Digit 9: 5.12%
3. Digit 7: 1.23%

Classification: 4
Confidence: 92.5%

Key Shape Features:
========================================

✓ Most important region: center
  (importance: 0.823)

✓ Dominant feature: Intersections
  (importance: 0.845)

Expected for '4':
  Looking for: intersection point, angled strokes
```

## 🎓 Educational Use

### Lesson Plan Ideas

**Lesson 1: Introduction to CNNs**
- Draw simple digits (1, 0)
- Show how model recognizes shapes
- Explain spatial importance (GradCAM)

**Lesson 2: Feature Recognition**
- Draw different digits
- Compare stroke features
- Understand geometric patterns

**Lesson 3: Model Limitations**
- Draw ambiguous digits
- See how confidence drops
- Understand why AI makes mistakes

**Lesson 4: XAI Concepts**
- Explain GradCAM vs Saliency
- Show regional analysis
- Discuss interpretability

## 🚀 Next Steps

After using the app:

1. **Explore patterns**: Try all digits 0-9
2. **Test edge cases**: Draw ambiguous digits
3. **Compare analyses**: See what distinguishes similar digits
4. **Read guide**: Check [SHAPE_INTERPRETATION_GUIDE.md](SHAPE_INTERPRETATION_GUIDE.md)
5. **Modify code**: Customize the app for your needs!

## 📚 Related Documentation

- [SHAPE_INTERPRETATION_GUIDE.md](SHAPE_INTERPRETATION_GUIDE.md) - Detailed shape analysis guide
- [README_MNIST_XAI.md](README_MNIST_XAI.md) - MNIST project documentation
- [START_HERE.md](START_HERE.md) - Main project guide

---

**Enjoy exploring how your handwritten digits are recognized by AI! ✏️🤖**
