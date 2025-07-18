# tradingutils

trading utils for penny-stock recommendations using technical analysis and ML-powered return predictions.

eventually, the goal is to combine both into a novel, terminal-like interface where users can choose stocks and minimalistically monitor prices/candles of the selected tickers, and trade using simple+effortless commands.

### instructions

[1] Set up a conda environment on any recent version of python (I use python 3.13).

[2] Install everything in requirements.txt, after activating the conda environment

[3] `brew install ta-lib` on Mac (or install TA-lib the right method for your OS)

[4] `cd` into the respective folder and mess with the code!

For the return-predictor, simply step through the .ipynb and tweak the presets as whimsy dictates. The fundamental model is a neural net with a small number of layers and nodes per layer. Using a Linear->BatchNorm->ReLU activation->Dropout scheme and AdamW+MSE+LR scheduler. The sentiment model is adapted from `finbert-tone` on HuggingFace. Yahoo Finance, Alpaca, and the Wayback Machine are all auxiliary resources leveraged by the code.

For the penny-stock-recommender, run `python3 penny-stock-recommender.py`. There are so many ways to use this code. First, make an Alpaca account and provide those API keys (from your _paper_ trading account). Create a Google Sheet shared to a service account from the GCP console, and provide the spreadsheet ID in the code. Or just comment out that code - it'll print to your terminal anyways. It's set to update every 10 minutes, but you can customize however you'd like - maybe even just run it once but put a cron job on the code within trading hours! Tweak the priors in the scoring system! It's meant to be a just-enough & customizable script that you can tailor to the signals you deem important in the securities that you're dealing with.
