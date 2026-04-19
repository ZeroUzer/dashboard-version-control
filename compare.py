import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import distance

class DashboardComparator:
    def __init__(self, threshold=0.1, use_preprocessing=True):
        self.threshold = threshold
        self.use_preprocessing = use_preprocessing

    def compare(self, image_path_1, image_path_2):
        image_1 = self._load_image(image_path_1)
        image_2 = self._load_image(image_path_2)

        if image_1 is None or image_2 is None:
            raise ValueError(f"Failed to load images: {image_path_1}, {image_path_2}")

        if self.use_preprocessing:
            image_1 = self._preprocess_image(image_1)
            image_2 = self._preprocess_image(image_2)

        height = max(image_1.shape[0], image_2.shape[0])
        width = max(image_1.shape[1], image_2.shape[1])

        image_1 = cv2.resize(image_1, (width, height))
        image_2 = cv2.resize(image_2, (width, height))

        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        similarity_score, difference_map = ssim(gray_1, gray_2, full=True)

        binary_diff = (difference_map < self.threshold).astype(np.uint8) * 255

        kernel = np.ones((3,3), np.uint8)
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel)
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)

        changed_pixels = np.sum(binary_diff > 0)
        total_pixels = binary_diff.size
        change_percentage = (changed_pixels / total_pixels) * 100

        color_changes = self._detect_color_changes(image_1, image_2, binary_diff)
        text_changes = self._detect_text_changes(image_1, image_2, binary_diff)
        position_changes = self._detect_position_changes(image_1, image_2)
        structural_changes = self._detect_structural_changes(image_1, image_2)

        colored_diff = cv2.applyColorMap((difference_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_image = cv2.addWeighted(image_1, 0.6, colored_diff, 0.4, 0)

        if similarity_score > 0.95:
            verdict = "Практически идентичны"
        elif similarity_score > 0.85:
            verdict = "Незначительные изменения"
        elif similarity_score > 0.70:
            verdict = "Заметные изменения"
        else:
            verdict = "Сильно отличаются"

        description = self._generate_detailed_description(
            change_percentage, similarity_score, color_changes,
            text_changes, position_changes, structural_changes
        )

        return {
            "similarity": round(similarity_score, 3),
            "change_percentage": round(change_percentage, 2),
            "difference_map": binary_diff,
            "overlay": overlay_image,
            "verdict": verdict,
            "description": description,
            "color_changes": color_changes,
            "text_changes": text_changes,
            "position_changes": position_changes,
            "structural_changes": structural_changes
        }

    def _load_image(self, image_path):
        with open(image_path, "rb") as file:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image

    def _preprocess_image(self, image):
        result = image.copy()
        result = cv2.GaussianBlur(result, (3, 3), 0)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        result = cv2.medianBlur(result, 3)
        return result

    def _detect_color_changes(self, image_1, image_2, diff_mask):
        changes = []
        
        hsv_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2HSV)
        hsv_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2HSV)
        
        diff_regions = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = diff_regions[0] if len(diff_regions) == 2 else diff_regions[1]
        
        significant_changes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            roi_hsv_1 = hsv_1[y:y+h, x:x+w]
            roi_hsv_2 = hsv_2[y:y+h, x:x+w]
            
            mean_hue_1 = np.mean(roi_hsv_1[:,:,0])
            mean_hue_2 = np.mean(roi_hsv_2[:,:,0])
            mean_sat_1 = np.mean(roi_hsv_1[:,:,1])
            mean_sat_2 = np.mean(roi_hsv_2[:,:,1])
            
            hue_diff = abs(mean_hue_1 - mean_hue_2)
            sat_diff = abs(mean_sat_1 - mean_sat_2)
            
            if hue_diff > 15:
                changes.append(f"изменился цвет элемента (оттенок изменился на {int(hue_diff)} градусов)")
                significant_changes += 1
            elif sat_diff > 30:
                changes.append(f"изменилась насыщенность цвета")
                significant_changes += 1
        
        if significant_changes > 0:
            return f"Обнаружено {significant_changes} цветовых изменений. " + "; ".join(set(changes[:3]))
        return None

    def _detect_text_changes(self, image_1, image_2, diff_mask):
        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        
        thresh_1 = cv2.adaptiveThreshold(gray_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh_2 = cv2.adaptiveThreshold(gray_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        text_diff = cv2.bitwise_and(thresh_1, thresh_2, mask=diff_mask)
        text_diff = cv2.bitwise_not(text_diff)
        
        contours, _ = cv2.findContours(text_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if 100 < area < 5000 and 0.2 < aspect_ratio < 10:
                text_regions += 1
        
        if text_regions > 0:
            return f"Вероятно, изменён текст в {text_regions} областях"
        return None

    def _detect_position_changes(self, image_1, image_2):
        orb = cv2.ORB_create(nfeatures=500)
        
        kp1, des1 = orb.detectAndCompute(image_1, None)
        kp2, des2 = orb.detectAndCompute(image_2, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            translation_x = M[0, 2]
            translation_y = M[1, 2]
            
            if abs(translation_x) > 10 or abs(translation_y) > 10:
                return f"Элементы сдвинуты (x: {int(translation_x)}px, y: {int(translation_y)}px)"
        
        return None

    def _detect_structural_changes(self, image_1, image_2):
        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        
        edges_1 = cv2.Canny(gray_1, 50, 150)
        edges_2 = cv2.Canny(gray_2, 50, 150)
        
        edge_diff = cv2.absdiff(edges_1, edges_2)
        edge_change = np.sum(edge_diff > 0) / edge_diff.size * 100
        
        if edge_change > 15:
            return "Существенно изменилась структура (добавлены или удалены графические элементы)"
        elif edge_change > 5:
            return "Изменилась структура расположения элементов"
        return None

    def _generate_detailed_description(self, change_percentage, similarity_score, 
                                        color_changes, text_changes, 
                                        position_changes, structural_changes):
        description_parts = []
        
        if change_percentage < 0.5:
            description_parts.append("Изменения отсутствуют или минимальны")
        elif change_percentage < 3:
            description_parts.append("Незначительные локальные изменения")
        elif change_percentage < 10:
            description_parts.append("Умеренные изменения")
        else:
            description_parts.append("Существенные изменения макета")
        
        if color_changes:
            description_parts.append(color_changes)
        
        if text_changes:
            description_parts.append(text_changes)
        
        if position_changes:
            description_parts.append(position_changes)
        
        if structural_changes:
            description_parts.append(structural_changes)
        
        if not (color_changes or text_changes or position_changes or structural_changes):
            if similarity_score > 0.92 and change_percentage < 3:
                description_parts.append("Визуальных отличий практически нет")
            elif similarity_score < 0.6:
                description_parts.append("Вероятно, это разные дашборды")
        
        return ". ".join(description_parts) + "."

    def save_overlay(self, overlay_image, output_path):
        success, encoded_image = cv2.imencode('.png', overlay_image)
        if success:
            with open(output_path, 'wb') as file:
                file.write(encoded_image.tobytes())


def run_quick_test(image_1_path, image_2_path):
    comparator = DashboardComparator()
    result = comparator.compare(image_1_path, image_2_path)

    print(f"Схожесть: {result['similarity']}")
    print(f"Изменено: {result['change_percentage']}%")
    print(f"Вердикт: {result['verdict']}")
    print(f"Описание: {result['description']}")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        run_quick_test(sys.argv[1], sys.argv[2])
    else:
        print("Использование: python compare.py <путь_к_картинке1> <путь_к_картинке2>")