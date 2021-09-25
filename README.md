# DeeplearningIneuron
Deep Learning with Ineuron



# Commonds for run in git 

```bash
git add . && git commit -m "docstring" && git push origin main
```


# Copy the path 
```bash
cp Research\ Notebook/demoo.ipynb .
```



# Replace code 

```bash
ctrl+h
```

## URL ADD
[git DeeplearningIneuron](https://github.com/it3037rakesh/DeeplearningIneuron.git)

## Images

![sample Image](plots/and.png)

![sample Image](plots/or.png)




```
def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(data)
    logging.info(f"This is actual dataframe{df}")
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename=modelName)
    save_plot(df, plotName, model)```