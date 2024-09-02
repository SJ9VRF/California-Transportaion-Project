# Bottom Line Separator Computation

This process computes the bottom line separator from a given point cloud file.

## Input
- **pc file:** A point cloud file.

## Output
- **Bottom Line Separator:** Equation representing the bottom line separator.

## Steps

### 1. Compute Center Part
- **Left Plane Separator:**
  - **Class:** `PointCloudClassifier`
  - **Input:** `pc`, class labels `1` and `3`
  - **Output:** Left plane separator equation (`eq_1`)

- **Right Plane Separator:**
  - **Class:** `PointCloudClassifier` (same class, different labels)
  - **Input:** `pc`, class labels `2` and `3`
  - **Output:** Right plane separator equation (`eq_2`)

- **Compute Points Between Hyperplanes:**
  - **Class:** `HyperplaneOperations`
  - **Input:** `pc`, `eq_1`, and `eq_2`
  - **Output:** Point cloud representing the center part (`pc_center`)

### 2. Compute Bottom Part
- **Class:** `PointCloudProcessor`
- **Input:** `pc`
- **Output:** Point cloud of the bottom part (`pc_bottom`)

### 3. Compute SVM Separator
- **Description:** Compute the separator between the gray label (`label = 0`) and the center label (`label = 3`).
- **Class:** `PointCloudClassifier`
- **Input:** `pc_bottom`, `label = 0`, `label = 3`
- **Output:** Bottom plane separator equation (`eq_3`)

### 4. Compute Intersection
- **Method 1:**
  - **Class:** `PlaneNearestPointsFitter`
  - **Input:** `eq_3`, `pc_bottom`
  - **Output:** Bottom fitted line equation (`bottom_fitted_line_eq`)

- **Method 2 (alternative approach):**
  - **Note:** Plane equation not given. This method involves coding an alternative approach for computing the intersection.

## Notes
- The process includes no normalization.
- Labels are retained throughout the computation.
