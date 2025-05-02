# OOD-AHM
directory for paper: 


Tite: 
Out-of-Distribution Detection with Attention Head Masking for Multi-modal Document Classification

Abstract:
Detecting out-of-distribution (OOD) data is critical for ensuring the reliability and safety of deployed machine learning systems by mitigating model overconfidence and misclassification. While existing OOD detection methods primarily
focus on uni-modal inputs, such as images or text, their effectiveness in multi-modal settings, particularly documents, remains underexplored. Moreover, most approaches prioritize decision mechanisms over optimizing the underlying dense 
embedding representations for optimal separation. In this work, we introduce Attention Head Masking (AHM), a novel technique applied to Transformer-based models for both uni-modal and multi-modal OOD detection. Our empirical results
demonstrate that AHM enhances embedding quality, significantly improving the separation between in-distribution and OOD data. Notably, our method reduces the false positive rate (FPR) by up to 10%, outperforming state-of-the-art approaches.
Furthermore, AHM generalizes effectively to multi-modal document data, where textual and visual information are jointly modeled within a Transformer architecture. To encourage further research in this area, we introduce FinanceDocs, a
high-quality, publicly available document AI dataset tailored for OOD detection. Our code1 and dataset2 are publicly available.
