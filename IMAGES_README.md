# Fertilizer Product Images - Setup Guide

## 📁 Folder Structure

The app expects images to be organized in the following structure:

```
Fertilizer Recommendation System/
├── images/
│   ├── Urea/
│   │   └── (Add your Urea fertilizer image here)
│   ├── DAP/
│   │   └── (Add your DAP fertilizer image here)
│   ├── NPK/
│   │   └── (Add your NPK fertilizer image here)
│   ├── SSP/
│   │   └── (Add your SSP fertilizer image here)
│   ├── MOP/
│   │   └── (Add your MOP fertilizer image here)
│   ├── Zinc Sulphate/
│   │   └── (Add your Zinc Sulphate fertilizer image here)
│   └── Compost/
│       └── (Add your Compost fertilizer image here)
├── app.py
├── train_model.py
└── (other files)
```

## 📸 Image Requirements

### Supported Formats
- JPG / JPEG
- PNG
- WEBP

### Recommended Specifications
- **Resolution:** 400×400 px to 1200×800 px
- **Format:** Product bag/package images
- **Size:** Less than 2MB per image
- **Background:** Clear, preferably white or transparent

### File Naming
You can name the image anything you want. Examples:
- `urea_bag.jpg`
- `fertilizer_urea.png`
- `product.jpg`

Just place ONE image file in each folder. The app will automatically detect and display it.

## 📋 All 7 Fertilizer Types

### 1. **Urea** (`images/Urea/`)
- **Strength:** High Nitrogen Fertilizer - 46% N
- **Characteristics:** White/cream colored crystalline granules
- **Search Keywords:** "Urea 50kg bag", "Urea fertilizer", "Nitrogen fertilizer", "Urea granules"
- **Common Brands:** Hindustan Urea Limited (HUL), KRIBHCO, Rashtriya
- **Image Example:** Fertilizer bag showing "46% N" or white granules
- **Priority:** ⭐⭐⭐ HIGH (Most commonly used)

### 2. **DAP** (`images/DAP/`)
- **Strength:** Diammonium Phosphate - 18% N, 46% P
- **Characteristics:** Grayish/brown granules/beads
- **Search Keywords:** "DAP fertilizer", "Diammonium phosphate bag", "DAP 50kg", "DAP gray granules"
- **Common Brands:** Hindustan Urea Limited, KRIBHCO, Boron, Rashtriya
- **Image Example:** Fertilizer bag with P and N markings, "DAP" clearly visible
- **Priority:** ⭐⭐⭐ HIGH (Primary phosphorus source)

### 3. **NPK** (`images/NPK/`)
- **Strength:** Balanced Compound Fertilizer - Nitrogen, Phosphorus, Potassium
- **Characteristics:** Mixed color granules (brown, pink, gray tones)
- **Search Keywords:** "NPK 10-26-26 fertilizer", "NPK compound", "NPK bag", "NPK balanced"
- **Common Brands:** KRIBHCO NPK, Rashtriya NPK, Novozymes, Chambal
- **Image Example:** Fertilizer bag showing NPK ratio (10:26:26, 12:32:16, 19:19:19)
- **Priority:** ⭐⭐⭐ HIGH (Most recommended for general use)

### 4. **SSP** (`images/SSP/`)
- **Strength:** Single Super Phosphate - 16% P
- **Characteristics:** Grayish/white granules, phosphorus-rich
- **Search Keywords:** "SSP fertilizer", "Single super phosphate", "SSP phosphate", "SSP 50kg"
- **Common Brands:** IFFC, Indian Farmers Fertilizers, Chambal, Rashtriya
- **Image Example:** Fertilizer bag marked "16% P" or "SSP" clearly labeled
- **Priority:** ⭐⭐ MEDIUM (For phosphorus deficiency correction)

### 5. **MOP** (`images/MOP/`)
- **Strength:** Muriate of Potash - 60% K2O (Potassium Oxide)
- **Characteristics:** White or reddish crystals/granules
- **Search Keywords:** "MOP fertilizer", "Muriate of potash", "MOP 50kg", "Potassium fertilizer"
- **Common Brands:** Karpco, KRIBHCO, Rashtriya, Indian Potash Limited
- **Image Example:** Fertilizer bag with "60% K2O" marked, white/red crystalline look
- **Priority:** ⭐⭐ MEDIUM (Potassium source for fruit/vegetable crops)

### 6. **Zinc Sulphate** (`images/Zinc Sulphate/`)
- **Strength:** Micronutrient Fertilizer - 21% Zn, 11% S
- **Characteristics:** Light blue or white crystalline powder/granules
- **Search Keywords:** "Zinc sulphate fertilizer", "ZnSO4 fertilizer", "micronutrient zinc", "Zinc powder"
- **Common Brands:** Century Enka, Ramakrishna Chemicals, Tronox, Agro Tech
- **Image Example:** Fertilizer package with "Zinc Sulphate" or "ZnSO4" clearly labeled
- **Priority:** ⭐⭐ MEDIUM (Prevents micronutrient deficiency in cereals/pulses)

### 7. **Compost** (`images/Compost/`)
- **Strength:** Organic Fertilizer - 0.5-2% N, Rich Organic Matter
- **Characteristics:** Dark brownish/black loose organic material, earthy smell
- **Search Keywords:** "Farm compost", "Vermicompost", "FYM compost", "Organic fertilizer", "Farm yard manure"
- **Common Brands:** Bio-gold, Gomukh, Organic Brands, Homemade FYM
- **Image Example:** Bag of compost/vermicompost, loose dark organic matter
- **Priority:** ⭐⭐⭐ HIGH (Soil conditioning, sustainable farming)

## 🎯 How to Add Images

### Step 1: Find and Download Images
Search online for each fertilizer type:

**Option A - Google Images:**
1. Go to Google Images
2. Search for each fertilizer (e.g., "Urea 50kg fertilizer bag")
3. Download high-quality product images
4. Ensure image rights are available for use

**Option B - Manufacturer/Retailer Websites:**
1. Visit fertilizer manufacturer websites (KRIBHCO, IFFCO, etc.)
2. Visit agricultural e-commerce sites
3. Download product photos

**Option C - Agriculture Portals:**
1. Government agriculture websites
2. State cooperative society websites
3. Agricultural research institute portals

### Step 2: Organize Your Downloads
1. Create a temporary folder for all downloaded images
2. Rename each image clearly:
   - `urea_bag.jpg`, `urea.jpg`, etc.
   - `dap_bag.jpg`, `dap.jpg`, etc.
   - `npk_10-26-26.jpg`, `npk.jpg`, etc.
   - `ssp_bag.jpg`, `ssp.jpg`, etc.
   - `mop_fertilizer.jpg`, `mop.jpg`, etc.
   - `zinc_sulphate.jpg`, `zinc.jpg`, etc.
   - `compost_organic.jpg`, `compost.jpg`, etc.

### Step 3: Copy Images to Folders
**Windows File Explorer Method:**

1. Press `Windows + R` to open Run dialog
2. Paste this path: `C:\Users\shrik\Desktop\Project\Fertilizer\images`
3. Press Enter (opens images folder with all subfolders)
4. For each fertilizer:
   - Open the subfolder (Urea, DAP, NPK, SSP, MOP, Zinc Sulphate, Compost)
   - Copy/paste your image file there
5. Repeat for all 7 folders

**PowerShell Method:**
```powershell
cd C:\Users\shrik\Desktop\Project\Fertilizer\images

# Copy images to their folders (example)
copy "C:\Users\YourUsername\Downloads\urea_bag.jpg" .\Urea\
copy "C:\Users\YourUsername\Downloads\dap_bag.jpg" .\DAP\
copy "C:\Users\YourUsername\Downloads\npk_bag.jpg" .\NPK\
copy "C:\Users\YourUsername\Downloads\ssp_bag.jpg" .\SSP\
copy "C:\Users\YourUsername\Downloads\mop_bag.jpg" .\MOP\
copy "C:\Users\YourUsername\Downloads\zinc_sulphate.jpg" ".\Zinc Sulphate\"
copy "C:\Users\YourUsername\Downloads\compost.jpg" .\Compost\
```

### Step 4: Verify All Folders Have Images
1. Open: `C:\Users\shrik\Desktop\Project\Fertilizer\images`
2. Check all 7 folders:
   - [ ] Urea/ - contains image
   - [ ] DAP/ - contains image
   - [ ] NPK/ - contains image
   - [ ] SSP/ - contains image
   - [ ] MOP/ - contains image
   - [ ] Zinc Sulphate/ - contains image
   - [ ] Compost/ - contains image

### Step 5: Test the App
1. Refresh browser: `Ctrl + R` (or go to http://localhost:8501)
2. Fill in soil test values:
   - Nitrogen (N): 50-150 ppm
   - Phosphorus (P): 20-80 ppm
   - Potassium (K): 50-200 ppm
   - pH: 5.5-8.5
   - Organic Matter: 0.5-2%
   - Select a Crop
3. Click "Analyze & Recommend"
4. The fertilizer images should now appear in the results! ✅

## 🔄 Workflow Example - Add Urea Image

**Complete step-by-step example for Urea:**

1. **Find Image:**
   - Go to Google Images
   - Search: "Urea 50kg fertilizer bag"
   - Find clear product photo of Urea bag
   - Right-click → Save image as → Name: `urea_bag.jpg`
   - Save to Downloads folder

2. **Copy to App Folder:**
   - Press Windows + R
   - Type: `C:\Users\shrik\Desktop\Project\Fertilizer\images\Urea`
   - Press Enter (opens Urea folder)
   - Drag & drop `urea_bag.jpg` here
   - Image now in correct folder ✅

3. **Test in App:**
   - Go to http://localhost:8501
   - Enter soil values: N=100, P=50, K=100, pH=6.5, OM=1.5%
   - Select crop: "Wheat" or "Rice"
   - Click "Analyze & Recommend"
   - If recommendation is "Urea", image appears below! 🎉

4. **Repeat for other 6 fertilizers**

## 🛠️ Missing Images Display

If an image is not found for a fertilizer recommendation:
- A placeholder box will appear saying:
  ```
  📸 [Fertilizer Name] Product Image
  Add image to: images/[FertilizerName]/
  Supported formats: JPG, PNG, JPEG, WEBP
  ```
- Simply add an image to that folder
- Refresh the browser to see the image

## 📝 Image Sources

### Recommended Sources:
1. **Government Websites**
   - KRIBHCO (Krishak Bharati Cooperative Limited)
   - State Agricultural Departments
   - ICAR (Indian Council of Agricultural Research)

2. **Fertilizer Manufacturers**
   - IFFCO
   - DOL (Department of Agriculture)
   - State cooperative societies

3. **Free Image Sites**
   - Unsplash
   - Pexels
   - Pixabay
   - Wikimedia Commons

4. **Agriculture Portals**
   - Agri-business websites
   - Farming communities
   - Agricultural education portals

## ✅ Complete Setup Checklist

- [ ] Created `images/` folder
- [ ] Created 7 subfolders:
  - [ ] images/Urea/
  - [ ] images/DAP/
  - [ ] images/NPK/
  - [ ] images/SSP/
  - [ ] images/MOP/
  - [ ] images/Zinc Sulphate/
  - [ ] images/Compost/
- [ ] Found 7 fertilizer product images
- [ ] Copied images to correct folders
- [ ] Each folder contains at least 1 image file
- [ ] Refreshed Streamlit app (Ctrl+R)
- [ ] Tested recommendations - images appear ✅
- [ ] All 7 fertilizer products have images ready

## 🎓 Tips for Best Results

1. **Use product bags:** Shows users what to look for in markets
2. **Clear images:** Avoid blurry or poorly-lit photos
3. **Consistent style:** Try to use similar quality images for all fertilizers
4. **Size appropriately:** Not too small (hard to see), not too large (slow loading)
5. **Real products:** Use actual fertilizer bags from shops/farms

## ❓ Troubleshooting

**Q: Image not showing even after I added it?**
A: 
- Restart the Streamlit app (Ctrl+C and re-run)
- Check file extension (.jpg, .png, .jpeg, .webp)
- Ensure image is in the correct folder
- Try refreshing the browser (Ctrl+R)

**Q: Which image displays if I have multiple?**
A: 
- The FIRST image in alphabetical order
- Rename to control: `001_`, `002_`, etc.

**Q: Can I use any size image?**
A: 
- Yes, but 400-1200px width works best
- Avoid very large files (>5MB)

**Q: Can I change folder names?**
A: 
- Not recommended - use: `Urea`, `DAP`, `NPK_Balanced`, `SSP` exactly
- Names must match fertilizer names in the app

## 🚀 Next Steps

1. Add your fertilizer product images to each folder
2. Restart the Streamlit app
3. Test by making recommendations
4. Images will automatically display in results!

---

**Questions?** Check that image folders have the correct names and contain image files in supported formats.
