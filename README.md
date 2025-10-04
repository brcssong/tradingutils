# tradingutils

**Update (10/3/2025)** Combining these tools with other strategies, I've been able to generate over a 20% return in the past 6 weeks in my personal account, compared to the S&P500's 5%! The starting capital was an amount between $XXXX and $XXXXXX. This wasn't a crazy lucky all-in play either; as the graph shows, the Sharpe seems pretty reasonable.

<img src="https://github.com/brcssong/tradingutils/blob/main/performance_update.jpg" height="200" /><img src="https://github.com/brcssong/tradingutils/blob/main/performance_graph.jpg" height="200" />

trading utils for high-intraday-momentum (penny-stock) recommendations using technical analysis and ML-powered return predictions.

eventually, the goal is to combine both into a novel, terminal-like interface where users can choose stocks and minimalistically monitor prices/candles of the selected tickers, and trade using simple+effortless commands.

i hope to also integrate this with an algorithmic flow that would allow for automation of trading, thus limiting possible risk exposure on the fastest-moving tickers.

### instructions

[1] Set up a conda environment on any recent version of python (I use python 3.13).

[2] Install everything in requirements.txt, after activating the conda environment

[3] `brew install ta-lib` on Mac (or install TA-lib the right method for your OS)

[4] `cd` into the respective folder and mess with the code!

### subproject specifities

For the return-predictor, simply step through the .ipynb and tweak the presets as whimsy dictates. The fundamental model is a neural net with a small number of layers and nodes per layer. Using a Linear->BatchNorm->ReLU activation->Dropout scheme and AdamW+MSE+LR scheduler. The sentiment model is adapted from `finbert-tone` on HuggingFace. Yahoo Finance, Alpaca, and the Wayback Machine are all auxiliary resources leveraged by the code.

For the penny-stock-recommender, run `python3 penny-stock-recommender.py`. There are so many ways to use this code. First, make an Alpaca account and provide those API keys (from your _paper_ trading account). Create a Google Sheet shared to a service account from the GCP console, and provide the spreadsheet ID in the code. Or just comment out that code - it'll print to your terminal anyways. It's set to update every 10 minutes, but you can customize however you'd like - maybe even just run it once but put a cron job on the code within trading hours! Tweak the priors in the scoring system! It's meant to be a just-enough & customizable script that you can tailor to the signals you deem important in the securities that you're dealing with.
