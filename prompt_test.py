import pandas as pd
import random
from tqdm import tqdm
import json
import time
from datasets import load_dataset,Dataset
import torch
import gradio as gr
import os
from sdmetrics.reports.single_table import QualityReport
import pickle
import gc
import asyncio
import subprocess
from sdmetrics.visualization import get_column_plot
from sdmetrics.reports.single_table import DiagnosticReport



num_gpu = torch.cuda.device_count()
class MLE_Efficieny:
    def __init__(self,original_df,synthetic_df):
        from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
        import numpy as np
        self.dataset=original_df
        self.synthetic_dataset=synthetic_df
        
        self.columns=self.dataset.columns
        self.target_column=self.dataset.columns[-1]
        self.target_values=None
        numerical_columns = self.dataset.select_dtypes(include=['int64', 'float64']).columns
        if(self.target_column in numerical_columns):
            self.evaluation="Regression"
        else:
            self.evaluation="Classification"
            self.target_values=self.dataset[self.target_column].unique()
        ordinal_encoder=LabelEncoder()
        self.dataset=self.dataset.apply(ordinal_encoder.fit_transform)
        self.synthetic_dataset=self.synthetic_dataset.apply(ordinal_encoder.fit_transform)
        
    def evaluate(self):
        if self.evaluation=="Regression":
            return self.regression_eval()
        else:
            return self.classification_eval()
            
    def regression_eval(self):
        print("Regression")
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor
        from lightgbm import LGBMRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics import mean_squared_error
        train_og,test_og=train_test_split(self.dataset,test_size=0.2)
        train_og_x=self.dataset.drop([self.target_column],axis =1)
        train_og_y=self.dataset[self.target_column]
        test_og_x=self.dataset.drop([self.target_column],axis =1)
        test_og_y=self.dataset[self.target_column]
        train_sy_x=self.synthetic_dataset.drop([self.target_column],axis =1)
        train_sy_y=self.synthetic_dataset[self.target_column]
        
        model1=ExtraTreesRegressor(n_estimators=100, random_state=0)
        model1.fit(train_og_x,train_og_y)
        model1_og=model1.score(test_og_x, test_og_y)
        del model1
        model1=ExtraTreesRegressor(n_estimators=100, random_state=0)
        model1.fit(train_sy_x, train_sy_y)
        model1_sy=model1.score(test_og_x, test_og_y)
        y_pred = model1.predict(test_og_x)
        y_true = test_og_y
        mse_1=mean_squared_error(y_true,y_pred)
        del model1

        model2=RandomForestRegressor(max_depth=2, random_state=0)
        model2.fit(train_og_x,train_og_y)
        model2_og=model2.score(test_og_x, test_og_y)
        del model2
        model2=RandomForestRegressor(max_depth=2, random_state=0)
        model2.fit(train_sy_x, train_sy_y)
        model2_sy=model2.score(test_og_x, test_og_y)
        y_pred = model2.predict(test_og_x)
        y_true = test_og_y
        mse_2=mean_squared_error(y_true,y_pred)
        del model2

        model3=LGBMRegressor()
        model3.fit(train_og_x,train_og_y)
        model3_og=model3.score(test_og_x, test_og_y)
        del model3
        model3=LGBMRegressor()
        model3.fit(train_sy_x, train_sy_y)
        model3_sy=model3.score(test_og_x, test_og_y)
        y_pred = model3.predict(test_og_x)
        y_true = test_og_y
        mse_3=mean_squared_error(y_true,y_pred)
        del model3

        model4=DecisionTreeRegressor()
        model4.fit(train_og_x,train_og_y)
        model4_og=model4.score(test_og_x, test_og_y)
        del model4
        model4=DecisionTreeRegressor()
        model4.fit(train_sy_x, train_sy_y)
        model4_sy=model4.score(test_og_x, test_og_y)
        y_pred = model4.predict(test_og_x)
        y_true = test_og_y
        mse_4=mean_squared_error(y_true,y_pred)
        del model4

        model5=KNeighborsRegressor()
        model5.fit(train_og_x,train_og_y)
        model5_og=model5.score(test_og_x, test_og_y)
        del model5
        model5=KNeighborsRegressor()
        model5.fit(train_sy_x, train_sy_y)
        model5_sy=model5.score(test_og_x, test_og_y)
        y_pred = model5.predict(test_og_x)
        y_true = test_og_y
        mse_5=mean_squared_error(y_true,y_pred)
        del model5

        model6=GradientBoostingRegressor()
        model6.fit(train_og_x,train_og_y)
        model6_og=model6.score(test_og_x, test_og_y)
        del model6
        model6=GradientBoostingRegressor()
        model6.fit(train_sy_x, train_sy_y)
        model6_sy=model6.score(test_og_x, test_og_y)
        y_pred = model6.predict(test_og_x)
        y_true = test_og_y
        mse_6=mean_squared_error(y_true,y_pred)
        del model6
        pd.options.plotting.backend = "plotly" 
        results = {
    "Model": ["ExtraTreesRegressor", "RandomForestRegressor", "LGBMRegressor", "DecisionTreeRegressor", "KNeighborsRegressor", "GradientBoostingRegressor"],
    "Original Dataset": [model1_og, model2_og, model3_og, model4_og, model5_og, model6_og],
    "Synthetic Dataset": [model1_sy, model2_sy, model3_sy, model4_sy, model5_sy, model6_sy]
        }
        temp = {
    "Model": ["ExtraTreesRegressor", "RandomForestRegressor", "LGBMRegressor", "DecisionTreeRegressor", "KNeighborsRegressor", "GradientBoostingRegressor"],
    "Mean Squared Error-Eval": [mse_1,mse_2,mse_3,mse_4,mse_5,mse_6]
        }
        temp_df=pd.DataFrame(temp)

        print(temp_df)
        
        temp_df.set_index('Model', inplace=True)  # Set 'Model' column as index for better visualization


        fig=temp_df.plot(kind='bar', title='Mean Squared Error of Regression Models')
                
        results_df = pd.DataFrame(results)
        return (results_df,fig)
   
    
    def classification_eval(self):
        print("Classification")
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier
        from lightgbm import LGBMClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss,ConfusionMatrixDisplay
        import plotly.graph_objs as go
        
        train_og,test_og=train_test_split(self.dataset,test_size=0.2)
        train_og_x=self.dataset.drop([self.target_column],axis =1)
        train_og_y=self.dataset[self.target_column]
        test_og_x=self.dataset.drop([self.target_column],axis =1)
        test_og_y=self.dataset[self.target_column]
        train_sy_x=self.synthetic_dataset.drop([self.target_column],axis =1)
        train_sy_y=self.synthetic_dataset[self.target_column]
        
        model1=ExtraTreesClassifier(n_estimators=100, random_state=0)
        model1.fit(train_og_x,train_og_y)
        model1_og=model1.score(test_og_x, test_og_y)
        del model1
        model1=ExtraTreesClassifier(n_estimators=100, random_state=0)
        model1.fit(train_sy_x, train_sy_y)
        model1_sy=model1.score(test_og_x, test_og_y)
        del model1

        model2=RandomForestClassifier(max_depth=2, random_state=0)
        model2.fit(train_og_x,train_og_y)
        model2_og=model2.score(test_og_x, test_og_y)
        del model2
        model2=RandomForestClassifier(max_depth=2, random_state=0)
        model2.fit(train_sy_x, train_sy_y)
        model2_sy=model2.score(test_og_x, test_og_y)
        del model2

        model3=LGBMClassifier()
        model3.fit(train_og_x,train_og_y)
        model3_og=model3.score(test_og_x, test_og_y)
        del model3
        model3=LGBMClassifier()
        model3.fit(train_sy_x, train_sy_y)
        model3_sy=model3.score(test_og_x, test_og_y)
        del model3

        model4=DecisionTreeClassifier()
        model4.fit(train_og_x,train_og_y)
        model4_og=model4.score(test_og_x, test_og_y)
        del model4
        model4=DecisionTreeClassifier()
        model4.fit(train_sy_x, train_sy_y)
        model4_sy=model4.score(test_og_x, test_og_y)
        del model4

        model5=KNeighborsClassifier()
        model5.fit(train_og_x,train_og_y)
        model5_og=model5.score(test_og_x, test_og_y)
        del model5
        model5=KNeighborsClassifier()
        model5.fit(train_sy_x, train_sy_y)
        model5_sy=model5.score(test_og_x, test_og_y)
        del model5

        model6=GradientBoostingClassifier()
        model6.fit(train_og_x,train_og_y)
        model6_og=model6.score(test_og_x, test_og_y)
        del model6
        model6=GradientBoostingClassifier()
        model6.fit(train_sy_x, train_sy_y)
        model6_sy=model6.score(test_og_x, test_og_y)
        y_pred = model6.predict(test_og_x)
        y_true = test_og_y
        pd.options.plotting.backend = "plotly"
        # fig=ConfusionMatrixDisplay.from_predictions(y_true, y_pred).figure_

        import plotly.figure_factory as ff
        import numpy as np
        z = confusion_matrix(y_true,y_pred)
        x=self.target_values
        y= self.target_values
        heatmap = go.Heatmap(z=z, x=x, y=y, colorscale='Blues')

        layout = go.Layout(title='Confusion Metrix')
        
        
        fig = go.Figure(data=[heatmap], layout=layout)
        
        
        fig.show()
                
        del model6

        results = {
    "Model": ["ExtraTreesClassifier", "RandomForestClassifier", "LGBMClassifier", "DecisionTreeClassifier", "KNeighborsClassifier", "GradientBoostingClassifier"],
    "Original Dataset": [model1_og, model2_og, model3_og, model4_og, model5_og, model6_og],
    "Synthetic Dataset": [model1_sy, model2_sy, model3_sy, model4_sy, model5_sy, model6_sy]
        }
        results_df = pd.DataFrame(results)
        return (results_df,fig)
        

        

 

class Prompt_test:
    def  __init__(self):
        self.llm=None
        self.synthetic_dataset=None
        self.dataset=None
        self.sampling_params=None
        self.model_name=None
        self.temperature=None
        self.top_p=None
        self.max_tokens=None
        self.dataset_type=None
        self.dataset_path=None
        self.split=None
        self.uploaded_file_csv=None
        self.uploaded_file_json=None
        self.synthetic_file_path=None
    async def load_model(self,model_name,temperature=0.7, top_p=0.9,max_tokens=2048):
        self.model_name=model_name
        self.temperature=temperature
        self.top_p=top_p
        self.max_tokens=max_tokens
        return "Model Configuration Stored"
        
    async def single_prompt(self,prompt_sys,prompt_user,prompt_assistant):
        prompt = f"[INST]<<SYS>>{prompt_sys}<</SYS>><<USER>>{prompt_user}<</USER>>[/INST]<<ASSISTANT>>{prompt_assistant}"
        #Run subprocess for single prompt
        input_prompt_path="temporary_prompt.txt"
        input_prompt_list=prompt.split("\n")
        f = open(input_prompt_path, "w")
        f.writelines(input_prompt_list)
        f.close()
        proc= await asyncio.create_subprocess_shell(" ".join(["conda","run","-n","vllm","python", "/src/vllm_inference.py","single_prompt",
                        "--model_name",self.model_name,
                        "--temperature",str(self.temperature),
                        "--top_p",str(self.top_p),
                        "--max_tokens",str(self.max_tokens), 
                        "--input_prompt_path",input_prompt_path
                        ]),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            error_message = stderr.decode() if stderr else 'Unknown error'
            raise gr.Error(f"Could not Generate Prompt: {error_message}")
        else:
            generated_out=stdout.decode()
            generated_text = generated_out.split("[OUT_TEXT]")[1].split("[/OUT_TEXT]")[0]
            hallucination = generated_out.split("[HALLUCINATION]")[1].split("[/HALLUCINATION]")[0]
            return generated_text, hallucination 
        
        
    async def generate_dataset(self, dataset_type, dataset_path, split, uploaded_file_csv, uploaded_file_json, num_samples_input, batch_size_input, input_prompt_sys,input_prompt_user, output_format,progress=gr.Progress(track_tqdm=True)):
        input_prompt= f"[INST]<<SYS>>{input_prompt_sys}<</SYS>><<USER>>{input_prompt_user}<</USER>>[/INST]"
        self.dataset_type=dataset_type
        self.dataset_path=dataset_path
        self.split=split
        self.uploaded_file_csv=uploaded_file_csv
        self.uploaded_file_json=uploaded_file_json
        self.synthetic_file_path=os.path.join("/src","SyntheticData.csv")
        #Run subprocess for generate_dataset
        input_prompt_path="temporary_prompt.txt"
        input_prompt_list=input_prompt.split("\n")
        f = open(input_prompt_path, "w")
        f.writelines(input_prompt_list)
        f.close()
        proc= await asyncio.create_subprocess_shell(" ".join(["conda","run","-n","vllm","python", "/src/vllm_inference.py","generate_dataset",
                        f"--model_name {self.model_name}",
                        f"--temperature {self.temperature}",
                        f"--top_p {self.top_p}",
                        f"--max_tokens {self.max_tokens}", 
                        f"--dataset_type {self.dataset_type}",
                        f"--dataset_path {self.dataset_path}",
                        f"--split {self.split}",
                        f"--uploaded_file_csv {self.uploaded_file_csv}",
                        f"--uploaded_file_json {self.uploaded_file_json}",
                        f"--num_samples_input {num_samples_input}",
                        f"--batch_size_input {batch_size_input}",
                        f"--input_prompt_path {input_prompt_path}", 
                        f"--output_format {output_format}"
                        ]),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            error_message = stderr.decode() if stderr else 'Unknown error'
            raise gr.Error(f"Could not Generate Prompt: {error_message}")
        # else:
        #     print(stdout)
        df=pd.read_csv(os.path.join("/src","SyntheticData.csv"))
        return df[:5],os.path.join("/src","SyntheticData.csv")
    async def evaluate(self):
        report = QualityReport()
        synthetic_dataset=pd.read_csv(self.synthetic_file_path)
        if self.dataset_type == 'HuggingFace':
            try:
                dataset=load_dataset(self.dataset_path,split=self.split)
            except:
                raise gr.Error('Unable To Load Dataset')
        elif self.dataset_type == 'CSV':
            try:
                dataset = load_dataset('csv', data_files =  self.uploaded_file_csv,split='train')
            except:
                raise gr.Error('Unable To Load Dataset')
        elif self.dataset_type == 'JSON':
            try:
                dataset = load_dataset('json', data_files = self.uploaded_file_json,split='train')
            except:
                raise gr.Error('Unable To Load Dataset')
               
        dataset=dataset.to_pandas()
        
        
        metadata = {"columns": {c: {"sdtype": "categorical" if dataset[c].dtype=='object' else 'numerical'} for c in dataset}}
        report.generate(dataset,synthetic_dataset ,metadata)
        
        fig1 = report.get_visualization(property_name='Column Shapes')
        fig2 = report.get_visualization(property_name='Column Pair Trends')
        

        fig3 = get_column_plot(
            real_data=dataset,
            synthetic_data=synthetic_dataset,
            column_name=dataset.columns[-1],
            plot_type='bar' if dataset[dataset.columns[-1]].dtype=='object' else 'distplot'
        )
        diag_report = DiagnosticReport()
        diag_report.generate(dataset, synthetic_dataset, metadata)
        fig4 = diag_report.get_visualization(property_name='Data Validity')
        results_df,fig6= MLE_Efficieny(dataset,synthetic_dataset).evaluate()
        #Create fig 5 for MLE efficiency
        pd.options.plotting.backend = "plotly"
        # Melting the DataFrame to make it suitable for plotting with Plotly through pandas
        melted_df = results_df.melt(id_vars=["Model"], var_name="Dataset", value_name="Score")
        
        # Plotting
        fig5 = melted_df.plot(x="Model", y="Score", color="Dataset", barmode='group',
                             title="Model Performance on Original vs Synthetic Datasets",
                             labels={
                                 "Score": "Score",
                                 "Model": "Model",
                                 "Dataset": "Dataset"
                             },
                             color_discrete_map={"Original Dataset": "blue", "Synthetic Dataset": "green"},
                             kind="bar",)
        
        # Customizing the layout for better readability
        fig5.update_layout(xaxis_tickangle=-45, xaxis={'categoryorder':'total descending'}) 
        
       
      
        return fig1,fig2,fig3,fig4,fig5,fig6
