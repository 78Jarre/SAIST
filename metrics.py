class Metrics:
    def __init__(self):
        self.intersection = self.union = self.matched = self.targets = 0
        self.false_area = self.pixels = 0

    def update(self, prediction, truth):
        import numpy as np
        from skimage import measure

        (prediction, truth) = (np.asarray(prediction, dtype=bool), np.asarray(truth, dtype=bool))
        if prediction.ndim != 2 or prediction.shape != truth.shape:
            raise ValueError("Expected prediction and mask with matching H x W shape")
        self.intersection += int(np.count_nonzero(prediction & truth))
        self.union += int(np.count_nonzero(prediction | truth))
        self.pixels += truth.size
        gt_props = measure.regionprops(measure.label(truth, connectivity=2))
        pred_props = measure.regionprops(measure.label(prediction, connectivity=2))
        self.targets += len(gt_props)
        used = set()
        for gt in gt_props:
            (best_idx, best_dist) = (-1, 3.0)
            for (index, pred) in enumerate(pred_props):
                if index in used:
                    continue
                distance = np.linalg.norm(np.array(pred.centroid) - np.array(gt.centroid))
                if distance < best_dist:
                    (best_idx, best_dist) = (index, distance)
            if best_idx != -1:
                used.add(best_idx)
        self.matched += len(used)
        self.false_area += sum(
            (int(pred.area) for (index, pred) in enumerate(pred_props) if index not in used)
        )

    def get(self):
        return dict(
            IoU=self.intersection / self.union if self.union else 1.0,
            Pd=self.matched / self.targets if self.targets else 0.0,
            Fa=self.false_area / self.pixels if self.pixels else 0.0,
        )
