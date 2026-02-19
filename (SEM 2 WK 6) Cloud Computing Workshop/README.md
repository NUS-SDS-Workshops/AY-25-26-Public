# Prerequisite
**[IMPORTANT]** 

Please install Azure CLI before joining the workshop! Visit [here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) to download the latest Azure CLI. 

Note that for **people running macOS machines**, you will need to **FIRST** install **homebrew** (which is a software package management system for macOS and Linux, it will be beneficial to have it :D). Follow [this link](https://brew.sh) to download. 

After Azure CLI installation, please create a new terminal and run 
```bash
az login
```
Then, you will be redirected to the Azure Portal in your web browser. Please login to the Azure for Student account that you have just created. 

Next, please navigate to your Azure Portal and look for the **Subscription** icon (looks like a yellow key). Navigate to find your subscription, i.e., **Azure for Students**, click into it and look for **Settings** > **Resource providers** on the left panel. 

Use the filter and type `Cdn` -> Click on round button to select `Microsoft.Cdn` -> Click **Register**.

Next, use the filter and type `policy` -> Click on round button to select `Microsoft.PolicyInsights` -> Click **Register**.


# Guide to Run Code
## Upload Data 
The following steps assume that you have your Azure portal and storage account created. Please refer to the slide page 20 if you haven't already done so. You should also have your `config.json` downloaded following the slide page 62. Move your `config.json` to the same directory where your `upload_data.ipynb` is. You should also follow the setup specified before **Step 0** to set the stage. 

### Step 1: 
Download the folder `data`. You should be able to see the three folders inside named `test`, `train_v1`, and `train_v2`. If you clone the repository, it should be in the same directory as your `config.json`. 

### Step 2: 
Navigate to the `upload_data.ipynb` file. Follow the comments and run the cells accordingly. Note that you should update some of the parameters based on your own file naming conventions.

## Train, Register, and Deploy Model
### Step 1: 
First, uncomment and run the top cell to ensure that you have the azure packages installed. Else, uncomment and run the subsequent cell to install that.

### Step 2:
To connect to MLClient, make sure that you have your `config.json` living in the same directory and replace the argument `file_name` to your actual config file. 

### Step 3: 
You can then proceed to run the next cell to create a compute instance. Note that this process will take about 3 minutes. 

### Step 4:
Once the cell has finished running, please proceed to run the 2 cells in **Section 2**. Note that this step assumes you have already uploaded the data as a DataAsset on Azure, following the tutorial before this. Then, you can follow the link the to your Azure Machine Learning portal to monitor the status of your job. It should take around 15 minutes. You have to go over to the Azure Machine Learning Studio to monitor the job progress. Only when the status shows `Completed`, you cna continue running the following cells. 

### Step 5: 
Copy the jobname you see from the `Name` column of your cell output and replace the variable `job_name` with yours. Then, run the cell and wait for a few seconds until it is done. 

### Step 6:
Run the 3 cells in **Section 4**. Then, once again navigate to your Azure Machine Learning portal > Endpoint and look for your endpoint. When you click into that particular endpoint, you should be able to see all the details, including the deployment state of your endpoint. Please stay and observe if the deployment is successful. 

### Step 7:
Note that this step assumes you have already uploaded the data as a DataAsset on Azure, following the tutorial before this. Also, we assume that the endpoint deployment is successful. Please copy the endpoint name created earlier with the format `fmnist-endpoint-********` and replace the variable `ENDPOINT_NAME` with it. Run the rest of the cells in the notebook and you have completed the tutorial on the ML Cycle: `Data Ingestion` > `Model Training` > `Model Registration` > `Model Deployment` > `Real-time Inference`. 

[IMPORTANT] Since this is submitting a job, please ensure that your Azure compute is on before running this section.

Congratulations! You have reached the end of this tutorial about **Using Cloud Platform as a Data Scientist**. Hope you enjoy the takeaways :)
