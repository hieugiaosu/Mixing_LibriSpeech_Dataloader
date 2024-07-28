from dataclasses import dataclass

@dataclass
class ErrorMessageUtil:
    only_support_batch_input = "we only support batch input. If you use a single input, please call .unsqueeze(0) before."
    complex_format_convert = "we require the input with input.size(1) == 2"
    two_input_in_the_same_shape = "2 input must be in the same shape"