try:
    from iris.nodes.segmentation import tensorrt_multilabel_segmentation, active_contour_dft_segmentation

    MultilabelSegmentation = tensorrt_multilabel_segmentation.TensorRTMultilabelSegmentation
    ActiveContourSegmentation = active_contour_dft_segmentation.ActiveContourDFTSegmentation
    
except ModuleNotFoundError:
    from iris.nodes.segmentation import onnx_multilabel_segmentation, active_contour_dft_segmentation

    MultilabelSegmentation = onnx_multilabel_segmentation.ONNXMultilabelSegmentation
    ActiveContourSegmentation = active_contour_dft_segmentation.ActiveContourDFTSegmentation
