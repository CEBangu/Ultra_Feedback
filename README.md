## What kind of responses does GPT-4 prefer? 

Open-Source driven DPO fine-tuning datasets often use GPT-4 to decide the "Accepted" response in DPO pairs. Thus, models fine-tuned on these datasets will learn to incorpoate GPT-4's preferences in their post-training responses.
In this study, we aimed to find a quickly computable and interpretable metric to characterize the types of responses that GPT-4 prefers. We then synthesized a new score, "LinearMetric" out of metrics that are maximized in the "Accepted" group, and ran 
Fine-Tuning experiments on Phi-3 (Microsoft), using the base model, a model fine-tuned with the orignial sorting of the data, and a model fine-tuned to maximize the LinearMetric. We evaluated the models via HuggingFace's (old) benchmarking tools.

We found that GPT-4 in general prefers responses that use more complex language. Moreover, we found that the fine-tuning Phi-3 on the dataset maximizing the LinearMetric improved perfomrance on the MMLU relative to baseline and the normal data split.

This project was my M1 Internship (1 day/week), and it was conducted at BunkaAI and the Institute Jean-Nicod. It was supervised by Dr. Charles DeDampierre  and Dr. Nicolas Baumard. 

The dataset used was the argilla/ultrafeedback-binarized-preferences-cleaned. They ask to cite the original: https://huggingface.co/datasets/openbmb/UltraFeedback. 

Finetuning was done using a script adapted from Maxime Labonne. The original: https://colab.research.google.com/drive/15iFBr1xWgztXvhrj5I9fBv20c7CFOPBE?usp=sharing
