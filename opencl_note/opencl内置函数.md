opencl内置函数：

size_t get_num_groups(uint dim)     返回dim指定维度上工作组的数目

size_t get_group_id(uint dim)            返回dim指定维度上工作组的ID

size_t get_local_size(uint dim)            返回工作组内dim指定维度上的工作项数目

size_t get_local_id(uint dim)                返回工作组内dim指定维度上的工作项ID