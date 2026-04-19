# Engineering Progress Report: OHE Arc Detection Computer Vision Pipeline

**Author:** Aditya Sharma  
**Project:** Overhead Equipment (OHE) & Pantograph Arc Detection  
**Current Status:** Version 5 (Stable / Data Collection Phase)

---

## 1. Executive Summary

This report details the iterative development of a computer vision pipeline designed to detect electrical arcs on railway Overhead Equipment (OHE) and pantographs. What began as a standard object detection task using YOLO evolved into a sophisticated 3-stage Hybrid Computer Vision Architecture. By combining deep learning with traditional OpenCV image processing, we successfully mitigated critical edge cases, including false positives ("ghosts") and undetected "super arcs."

Currently, we have deployed Version 5 (V5). While it is our most accurate model, detecting significantly more arcs than standard OHE hardware reports, active model training has been temporarily paused to focus on data collection. This will prevent model overfitting and pave the way for a highly optimized edge-deployment architecture (Version 6).

---

## 2. Phase 1: Initial YOLO Training & The "Ghost" Problem

Our initial pipeline started with processing videos into a dataset of frames for labeling and training a YOLO model. However, the model immediately suffered from a high rate of false positives—hallucinating "ghosts" in the footage. Root cause analysis indicated that the dataset contained an overwhelming number of frames without arcs, skewing the model.

### The Solution: Software CROI Masking & Dataset Curation
To restrict the model's focus, we introduced a software-based Central Region of Interest (CROI) mask, geometrically defined as a polygon within a rectangular stencil.

* **Color Iteration:** We discovered that a standard black mask caused the model to continue hallucinating ghosts. Switching the mask to a **cyan color** yielded immediate and drastic improvements, boosting our accuracy to **88% true positives**.

Using this stabilized model, we performed a massive data cleanup:
* Extracted clean frames of actual arcs and pantographs.
* Conducted manual sorting of the images as a quality check.
* Realized the old model's dataset was highly corrupted, which was the primary reason for the ghosts.
* **Retraining:** We trained a new model on the collected data with a much better ratio: **288 arcs + 123 non-arcs**.
* **Strategy:** We utilized a 3-folder structure plus negative sampling (no labels). Notably, we specifically classified pantographs as arcs during labeling to cover edge cases and train the model more robustly.

**Phase 1 Result:** The new model yielded zero false positives and only identified actual arcs. Because the model improved so much, we were able to increase the size of the ROI stencil, with the long-term goal of improving the model until the mask is no longer needed.

---

## 3. Phase 2: The "Big Arc" Blindspot & Hardware Report Discrepancies

We conducted a report comparison using our stabilized model (Version 2) against physical OHE hardware reports.
* **OHE Report:** 208 arcs detected.
* **AI (V2) Report:** 570+ arcs detected.

While it was a positive sign that our AI caught a majority of "mini arcs" that hardware missed, the OHE reports revealed a critical failure: **V2 was completely blind to "Big" and "Huge" arcs.** We concluded that the ROI masking was inadvertently obscuring these massive flashes.

### Image Processing & Data Gathering
We wrote a script to extract images of the big arcs specifically from the OHE reports. We found these images often featured a "black ring" artifact, rendering them impossible to use for direct AI training. Instead, we used OpenCV to process them and gather metrics to create rule-based detection logic:
* **Frame Coverage by Arc:** Average: 37.31% | Small: 9.58% | Largest: 97.60%
* **Color Profile:** Average Hue: 88.12 | Average Saturation: 17.97 | Average Brightness: 253.33 (Resulting in a super bright, blue/almost-white tint).

*Note: We also observed the bottom 25% of the screen never contains arcs, so we created a mask for that specific region to preemptively prevent ghost sightings.*

---

## 4. Phase 3: The 3-Stage Hybrid Architecture

Using the localization data and intensity gradients (>0.5) from the big arcs, we designed a **Hybrid Computer Vision Architecture** that splits processing into three distinct stages:

1. **Stage 1 (Basic Arcs):** Standard YOLO detection utilizing the optimized polygon ROI mask for standard-sized arcs.
2. **Stage 2 (Big Arcs):** OpenCV-driven processing using a circular/lenient mask based on the localization data to capture arcs that fall outside the Stage 1 polygon.
3. **Stage 3 (Huge/Auto Arcs):** Hardcoded thresholding. If a massive arc makes recognizing impossible by turning the screen white, the system automatically accepts it as an arc if **more than 40-50% of the screen in any orientation** matches the specific color profile described above. This logic was manually checked against the video and verified as true positives.

---

## 5. Phase 4: The "Teacher-Student" Dataset & Lens Flares

Using V2 on arcs and backgrounds, along with carrying over V2 weights, we created an entirely new "Teacher-Student" dataset: **881 arcs and 300 non-arcs**.

However, a new issue emerged: **Lens Flares**. The model learned that lens flares were visually easier to recognize than the arcs themselves, leading to false detections.

### Attempted Mitigations for Flares:
1. **Mathematical Pre-processing:** We attempted to use scripts to mathematically locate and apply pre-processing masks to the flares. *Result:* This dropped our FPS by 50% (halving our speed), and the flares were too random to detect accurately. This approach was dropped.
2. **Future Hardware Fix:** We will attempt physical filtering of flares (via color, size, etc.) directly on the camera lens.
3. **Future Software Fix:** Implement rigorous negative sampling for flares after manual sorting.

---

## 6. Phase 5: V3/V4 Regressions and Mitigation Attempts

While the dataset was running, the new models (V3 & V4) showed severe regressions. The report started seeing ghosts over the OCR (Optical Character Recognition) region, and Stage 1 failed to find proper arcs even at a 65% confidence level.

**Troubleshooting Steps Taken:**
* Placed a permanent mask over the OCR region. *Result:* Still saw ghosts.
* To mitigate these negative changes, we:
  * Reduced the Super Arc screen coverage threshold from 50% to 40%.
  * Allowed for whiter flashes in the Super Arc detection logic.
  * Removed the OCR cover entirely, as it provided no benefit.
* Tested YOLO confidence thresholds: 0.365 yielded "absolute garbage," while 0.75 was "still okay."

We hypothesized the dataset might be wrong. We remade the dataset using V2, explicitly **dropping the pantograph class** during labeling, and trained Version 4. To help the reporting, we decreased the blocked area in stages:
* Dropped Stage 2 size limit by 75%.
* Removed shape determination math.
* Made colors more lenient: Hue (50-140) + Saturation (>80).
* *Result:* It still didn't work.

---

## 7. Phase 6: Root Cause Analysis & Version 5

We asked: *Why did V2 succeed while the exact same methods failed for newer versions?*
Testing V2 with the latest reporting script confirmed V2 still worked perfectly. We identified two core problems:
1. **Labeling:** V2 utilized autodistill for labeling. The manual labeling in V3/V4 only affected places where the old model failed. This was the most basic problem.
2. **Weights:** V2 did not transfer weights from previous iterations, whereas V3/V4 carried over corrupted weights.

### Deployment of Version 5: 
By returning to autodistill for labeling and utilizing V2 weights, we generated Version 5.
* **Result:** V5 is our "Best model as of yet." It acts as an amplified V2, catching a lot of the OHE report arcs and many more.
* **Issue:** Ghosts reappeared, partially due to the less restrictive masking we had implemented.
* **Current Solution:** We have stopped active development on V5 to avoid overfitting. We will extract frames using V5, manually sort for ghost detects, and apply negative sampling on a future model. This can be done safely once we gather new data. Furthermore, hardware integration in the future will give us more of an edge.

---

## 8. Strategic Roadmap: Version 6 & Teacher-Student Distillation

As we move toward final edge deployment (Version 6), we will use autodistill datasets alongside completely **fresh weights**.

To maximize accuracy and efficiency, we will adopt a Teacher-Student model distillation approach:
1. **The "Teacher" (Heavy Model):** We will train a computationally heavy model overnight using fresh (or CV2) weights on our current dataset. This model will not run on the edge; instead, it will be used offline to extract data and create a pristine report (representative of future edge reports).
2. **The "Student" (Lightweight Model):** Once we have more data and compute, we will use the heavy model's final, perfect dataset to train a highly optimized, lightweight model tailored specifically for real-world edge hardware. This provides the "best of both worlds."