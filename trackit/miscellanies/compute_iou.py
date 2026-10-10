def compute_iou(rec1, rec2):
    
    S_rec1 = rec1[2] * rec1[3]
    S_rec2 = rec2[2] * rec2[3]

    
    sum_area = S_rec1 + S_rec2

    
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[1] + rec1[3], rec2[1] + rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[0] + rec1[2], rec2[0] + rec2[2])

    
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
