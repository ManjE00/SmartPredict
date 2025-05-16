from setuptools import setup, find_packages

setup(
    name="sales_forecast",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.28.0',
        'pandas>=2.1.0',
        'scikit-learn>=1.3.0',
        'plotly>=5.15.0',
        'holidays>=0.28',
        'joblib>=1.3.1',
        'xgboost>=2.0.0'
    ],
)