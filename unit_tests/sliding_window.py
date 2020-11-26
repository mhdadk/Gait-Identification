index_ranges = []
window_overlap = 2/3
window_length = 100
test_length = 500

# ((test_length - window_length) / (window_length*(1-window_overlap))) + 1

last = ((test_length - window_length) / (window_length*(1-window_overlap))) + 1

for i in range(int(last)):
    start = int(i*window_length*(1-window_overlap))
    end = int((i*window_length*(1-window_overlap)) + window_length)
    index_ranges.append((start,end))

# iter_length = test_length

# # (test_length*overlap/window_length) + 1/window_overlap

# for i in range(3):
#     start = int(i*window_length*(1-window_overlap))
#     end = int((i*window_length*(1-window_overlap)) + window_length)
#     index_ranges.append((start,end))