def get_direction(centroids, frame_shape):
    
    if len(centroids) < 10:
        return 'inconclusive'
    
    # compare_sum_of_each_half(centroids)
    neighbor_direction(centroids)

def neighbor_direction(centroids):
    x_list, y_list = get_x_y_lists(centroids)
    y_sum = x_sum = 0
    for i in range(0, len(centroids)-1):
        x_sum += x_list[i]-x_list[i+1]
        y_sum += y_list[i]-y_list[i+1]
    if x_sum > 0:
        return "left"
    else:
        return "right"

def compare_sum_of_each_half(centroids):
    mid_point = len(centroids)//2
    x_list, y_list = get_x_y_lists(centroids)

    x_list_first, x_list_last = split_list(x_list)
    y_list_first, y_list_last = split_list(y_list)

    if x_list_first > x_list_last:
        return "left"
    else:
        return "right"

def split_list(a_list):
    half = len(a_list)//2
    return sum(a_list[:half]), sum(a_list[half:])

def get_x_y_lists(centroids):
    x_list, y_list = [], []

    for centroid in centroids:
        x_list.append(centroid[0])
        y_list.append(centroid[1])

    return x_list, y_list