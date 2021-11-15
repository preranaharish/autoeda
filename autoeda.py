import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import io 
import warnings
from PIL import Image
image = Image.open('logo.png')
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 12})

others = [] 
class DataFrame_Loader():

    
    def __init__(self):
        
        print("Loading DataFrame")
        
    def read_csv(self,data):
        self.df = pd.read_csv(data)
        return self.df

def main():
    st.image(image, use_column_width=True)
    new_title = '<p style="font-family:Brush Script MT; color:Green; font-size: 32px;">Keep Calm and just write Inferences.... </p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.header("Exploratory Data Analysis")
    data = st.file_uploader("Upload a Dataset", type=["csv"])
    if data is not None:
        df = load.read_csv(data)
        st.dataframe(df.head())
        st.success("Data Frame Loaded successfully")

        st.subheader("Shape of the Dataset")
        text = "The dataset has "+str(df.shape[0])+" rows and "+str(df.shape[1])+" columns"
        st.write(text)

        st.subheader("5-point Summary")
        st.write(df.describe())

        st.subheader("Data types information")
        dtypes = []
        for col in df.columns:
            dtypes.append(df[col].dtype)
        dtype_info = pd.DataFrame({'Column':df.columns,'Data type':dtypes})
        dtype_info = dtype_info.astype('str')
        st.dataframe(dtype_info)
        

        st.subheader("Missing Values")
        st.dataframe(df.isna().sum())

        cat_cols = df.select_dtypes('object').columns
        num_cols = df.select_dtypes(np.number).columns

        st.subheader("Do you want to drop any column/columns?")
        all_columns_names = df.columns
        selected_columns_names = st.multiselect("Select Column/columns to drop ",all_columns_names)
        st.warning("This is a permanent Change")
        if st.checkbox("Drop Selected Column"):
            df.drop(selected_columns_names,inplace=True,axis=1)
            st.success("Column dropped Successfully")
            all_columns_names = df.columns
            st.subheader("New Set of columns are")
            st.dataframe(df.columns)

            st.subheader("Categorical Columns")
            cat_cols = df.select_dtypes('object').columns
            st.write(cat_cols)

            st.subheader("Numerical Columns")
            num_cols = df.select_dtypes(np.number).columns
            st.write(num_cols)


        st.subheader("Do you want to type cast any of the columns?")
        status = st.radio(label="",options=('No', 'Yes'))
        if (status == 'No'):
            st.success("No")
        else:
            st.success("Yes")
            for i,col in enumerate(cat_cols):
                st.write(col)
                cast = st.radio("Select type: ", ('None', 'Numerical'),key=i)
                if (cast == 'None'):
                    pass
                else:
                    df[col] = df[col].astype(np.number)
            for i,col in enumerate(num_cols):
                st.write(col)
                cast = st.radio("Select type: ", ('None', 'Categorical'),key=i)
                if (cast == 'None'):
                    pass
                else:
                    df[col] = df[col].astype('object')

            st.subheader("Categorical Columns")
            cat_cols = df.select_dtypes('object').columns
            st.write(cat_cols)

            st.subheader("Numerical Columns")
            num_cols = df.select_dtypes(np.number).columns
            st.write(num_cols)
        
        st.subheader("Value Counts of Categorical columns")
        for col in cat_cols:
            st.write(df[col].value_counts())

        st.subheader("Treat Missing values")
        x=df.isna().sum()
        y=df.isna().sum()/len(df)*100
        null_c=pd.concat([x,y],axis=1,keys=['null_value_sum','null_value_percentage'])
        miss_cols = null_c[null_c['null_value_sum']>0].index
        for i,col in enumerate(miss_cols):
            text = "Impute null values of "+str(col)
            if st.checkbox(text):
                st.warning("This is a permanent Change")
                status = st.radio(label="",options=('Drop Missing Values', 
                'Mean Imputation',
                'Median Imputation',
                'Mode Imputation',
                'Custom Value Imputation','Drop Column'),key=i)

                if status == 'Custom Value Imputation':
                    user_input = st.text_input("Enter value:",key = i)
                if st.checkbox("Impute",key=i):
                    if (status == 'Drop Missing Values'):
                        df.dropna(subset=[col],inplace=True)
                    elif (status == 'Mean Imputation'):
                        df[col] = df[col].replace(np.nan,df[col].mean())

                    elif (status == 'Median Imputation'):
                        df[col] = df[col].replace(np.nan,df[col].median())

                    elif (status == 'Mode Imputation'):
                        df[col] = df[col].replace(np.nan,df[col].mode())

                    elif (status == 'Drop Column'):
                        df = df.drop(col,axis=1)

                    elif (status == 'Custom Value Imputation'):
                        df[col] = df[col].replace(np.nan,user_input)
                        
                    else:
                        pass
                    st.dataframe(df.head())
        
        st.subheader("Missing Values")
        st.dataframe(df.isna().sum())
        


        st.subheader("Skewness")
        st.dataframe(df.skew())

        st.subheader("Select the target")
        target = st.selectbox("Select target attribute ",all_columns_names)
        if st.checkbox("Confirm this attribute as the Target"):
            text = target+" is set as the target"
            st.success(text)
            global others 
            others = list(df.columns)
            others.remove(target)
        else:
            target = None
            st.error("No target attribute selected")

        st.subheader("Univariate Analysis of categorical data")
        plots = ["Count plot","Pie chart","Bar plot"]
        plot = st.selectbox("Select plot type ",plots)
        if st.checkbox("Generate plots",key='cat'):
            if plot == 'Count plot':
                for col in cat_cols:
                    plt.figure(figsize=(18,5))
                    plt.title(col)
                    st.write(sns.countplot(df[col]))
                    st.pyplot()
                    
            elif plot == 'Pie chart':
                for col in cat_cols:
                    plt.figure(figsize=(18,5))
                    plt.title(col)
                    st.write(plt.pie(df[col].value_counts(),labels=df[col].value_counts().index,autopct='%1.2f%%'))
                    st.pyplot()
            else:
                for col in cat_cols:
                    plt.figure(figsize=(18,5))
                    plt.title(col)
                    st.write(df[col].value_counts().plot(kind='bar'))
                    st.pyplot()

        st.subheader("Univariate Analysis of numerical data")
        plots = ["striplot","swarmplot","violinplot","distplot","boxplot","histogram"]
        plot = st.selectbox("Select plot type ",plots)
        if st.checkbox("Generate plots",key='num'):
            if plot == "striplot":
                for col in num_cols:
                    plt.figure(figsize=(18,5))
                    plt.title(col)
                    st.write(sns.stripplot(df[col]))
                    st.pyplot()
            
            elif plot == "swarmplot":
                for col in num_cols:
                    plt.figure(figsize=(18,5))
                    plt.title(col)
                    st.write(sns.swarmplot(df[col]))
                    st.pyplot()
            
            elif plot == "violinplot":
                for col in num_cols:
                    plt.figure(figsize=(18,5))
                    plt.title(col)
                    st.write(sns.violinplot(df[col]))
                    st.pyplot()

            elif plot == "distplot":
                for col in num_cols:
                    plt.figure(figsize=(18,5))
                    plt.title(col)
                    st.write(sns.distplot(df[col]))
                    st.pyplot()


            elif plot == 'histogram':
                for col in num_cols:
                    plt.figure(figsize=(18,5))
                    plt.title(col)
                    st.write(df[col].plot(kind='hist'))
                    st.pyplot()
            
            if plot == "boxplot":
                for col in num_cols:
                    plt.figure(figsize=(18,5))
                    plt.title(col)
                    st.write(sns.boxplot(df[col]))
                    st.pyplot()
            
            else:
                pass   
        st.header("Bivariate Analysis")
        
        if st.checkbox("Generate plots",key='bi_plots'):
            st.subheader("Bivariate Analysis with respect to target")
            if df[target].dtype == 'object':
                for col in cat_cols:
                    if col != target:
                        plt.figure(figsize=(18,5))
                        st.write(sns.catplot(data=df,col=col,x=target,kind='count'))
                        st.pyplot()
                for col in num_cols:
                    if col != target:
                        plt.figure(figsize=(18,5))
                        st.write(sns.countplot(df[col],hue=df[target]))
                        st.pyplot()
            
            if df[target].dtype == np.number:
                for col in cat_cols:
                    if col != target:
                        plt.figure(figsize=(18,5))
                        st.write(sns.barplot(x=df[col],y=df[target]))
                        st.pyplot()
                for col in num_cols:
                    if col != target:
                        plt.figure(figsize=(18,5))
                        st.write(sns.scatterplot(df[col],df[target]))
                        st.pyplot()

            st.subheader("EDA with respect to other cols")       
            attr = st.selectbox("Select attribute ",others)
            if st.checkbox("Generate plots",key='attr'):
                if df[attr].dtype=='object':
                    for col in cat_cols:
                        if col != attr:
                            plt.figure(figsize=(18,5))
                            st.write(sns.catplot(data=df,col=col,x=attr,kind='count'))
                            st.pyplot()
                    for col in num_cols:
                        if col != attr:
                            plt.figure(figsize=(18,5))
                            st.write(sns.barplot(y=df[col],x=df[attr]))
                            st.pyplot()

                if df[attr].dtype == np.number:
                    for col in cat_cols:
                        if col != attr:
                            plt.figure(figsize=(18,5))
                            st.write(sns.barplot(x=df[col],y=df[attr]))
                            st.pyplot()
                    for col in num_cols:
                        if col != attr:
                            plt.figure(figsize=(18,5))
                            st.write(sns.scatterplot(df[col],df[attr]))
                            st.pyplot()
            st.subheader("Pair Plot")
            st.write(sns.pairplot(df))
            st.pyplot()

            st.subheader("Correlation")
            st.write(sns.heatmap(df.corr(),annot=True))
            st.pyplot()

            

if __name__ == '__main__':
    load = DataFrame_Loader()
    main()