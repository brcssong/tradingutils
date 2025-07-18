import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass, AssetExchange

ALPACA_API_KEY = '--API KEY HERE--'
ALPACA_API_SECRET = '--API SECRET HERE--'

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

max_price_stock = 4
min_price_stock = 0.5
min_vol = 50

def rsi(series, period=3):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = -delta.where(delta < 0, 0).rolling(period).mean()

    rs = gain / loss
    rs = rs.where(loss != 0, np.inf)

    return 100 - (100 / (1 + rs))

def calculate_vwap(df):
    tp = (df['h'] + df['l'] + df['c']) / 3
    return (df['v'] * tp).cumsum() / df['v'].cumsum()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def analyze_stock(symbol, runup_thresh=15.0):
    pst = pytz.timezone('US/Pacific') # CHANGE TO YOUR TIME ZONE
    now = pst.localize(datetime.now())
    start = now - timedelta(minutes=15)

    start_utc = start.astimezone(pytz.UTC).isoformat()
    now_utc = now.astimezone(pytz.UTC).isoformat()

    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start_utc,
        end=now_utc,
        feed='iex', # `sip` is better if you have an Alpaca premium plan
        limit=15
    )

    bars = data_client.get_stock_bars(req).df
    if bars.empty or len(bars) < 5:
        return None

    df = bars.reset_index(drop=True)
    df['EMA3'] = ema(df['close'], span=3)
    df['EMA8'] = ema(df['close'], span=8)
    df['RSI3'] = rsi(df['close'], period=3)
    tmp = df.rename(columns={'high':'h','low':'l','close':'c','volume':'v'})
    df['VWAP'] = calculate_vwap(tmp)

    open_price  = df['open'].iloc[0]
    last_price  = df['close'].iloc[-1]
    if last_price > max_price_stock:
        return None
    if open_price == 0:
        return None
    
    percent_change = (last_price - open_price) / open_price * 100
    runup_penalty = max(0, (percent_change - runup_thresh) / runup_thresh)

    # Does first EMA3 > EMA8 cross in window?
    cross_up = (df['EMA3'].shift(1) <= df['EMA8'].shift(1)) & (df['EMA3'] > df['EMA8'])

    # Does the RSI cross 50?
    rsi = df['RSI3']
    rsi_cross = (rsi.shift(1) < 50) & (rsi > 50)

    # Is the stock going on a breakout run?
    prior_high = df['high'].iloc[:-1].rolling(10, min_periods=1).max().iloc[-2]
    distance_to_high = (last_price - prior_high) / prior_high
    breakout_run = np.tanh(distance_to_high * 10)

    # Has the stock dipped recently? Score this inversely to anticipate a potential rebound.
    min_price = df['low'].min()
    dip_pct   = max(0, (open_price - min_price) / open_price) if open_price else 0
    dip_score = np.tanh(dip_pct * 5)

    # Build result in the DataFrame. Used for scoring.
    total_vol = df['volume'].sum()
    res = {
        'symbol': symbol,
        'last_price': last_price,
        'percent_change': max(-8, percent_change),
        'volume': total_vol,
        'RSI3': rsi.iloc[-1],
        'EMA3': df['EMA3'].iloc[-1],
        'EMA8': df['EMA8'].iloc[-1],
        'VWAP': df['VWAP'].iloc[-1],
        'first_ema_cross': cross_up.any(),
        'rsi_cross50': rsi_cross.any(),
        'breakout_run': breakout_run,
        'runup_penalty': runup_penalty,
        'dip_score': dip_score,
    }

    return res

def get_penny_stocks(
    min_price: float = min_price_stock,
    max_price: float = max_price_stock,
    min_volume: int = min_vol
) -> list[str]:
    assets = trading_client.get_all_assets(
        GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass.US_EQUITY,
            exchange=AssetExchange.NASDAQ # Change (or remove) if wish to expand stock search.
        )
    )
    np.random.shuffle(assets)
    symbols = []
    for a in assets:
        if not a.tradable:
            continue
        req = StockLatestBarRequest(
            symbol_or_symbols=a.symbol,
            feed='iex' # again, OR 'sip'
        )
        cl = data_client.get_stock_latest_bar(req)
        if a.symbol not in cl:
            continue
        bar = cl[a.symbol]
        close, vol = bar.close, bar.volume
        if min_price <= close <= max_price and vol >= min_volume:
            symbols.append(a.symbol)
            print('Updated size of ticker list to', len(symbols))
        time.sleep(0.3) # skirt Alpaca API rate limits
    print('Final ticker list:', symbols)
    return symbols

# Tweak this based off of the `get_penny_stocks(...)` result! I got this from my own run.
symbols = ['PCSA', 'BTAI', 'BNZI', 'MDCXW', 'DTI', 'INBS', 'NVX', 'CBUS', 'SLRX', 'UK', 'MRNOW', 'IOTR', 'WGRX', 'TC', 'NRSN', 'TLRY', 'ONCY', 'MAMO', 'KLTOW', 'KBSX', 'BNAI', 'YGMZ', 'WTO', 'MVSTW', 'ANGH', 'NKTX', 'SEAT', 'RVYL', 'MESA', 'POLEW', 'IHRT', 'PALI', 'QETAR', 'NESRW', 'ANEB', 'MNDO', 'ENLV', 'MAPS', 'WETO', 'RDACR', 'FLDDW', 'FBGL', 'LSB', 'VSA', 'LNKS', 'CPOP', 'ALXO', 'JDZG', 'SPWR', 'SBFMW', 'SCWO', 'GIGM', 'PTIXW', 'NNDM', 'DRIO', 'XBP', 'ANTX', 'TRSG', 'TRUE', 'MODD', 'FRGT', 'NTCL', 'FTEK', 'CIIT', 'NXL', 'HKIT', 'WAFU', 'CTNT', 'IPCXR', 'ARTW', 'CLIK', 'AUROW', 'CYCU', 'RPTX', 'DRMA', 'AZI', 'AXTI', 'SLS', 'BTMWW', 'CNTY', 'SCKT', 'ASPCR', 'SABS', 'WLDSW', 'IBIO', 'ONMD', 'GMHS', 'EDSA', 'IBG', 'WNW', 'IMG', 'BCLI', 'ALEC', 'LXRX', 'SVIIW', 'PRPH', 'SIDU', 'ASNS', 'NXGLW', 'BOLD', 'BLDP', 'MNOV', 'INVZ', 'VFSWW', 'NCI', 'MGIH', 'SND', 'ADTX', 'RIBBR', 'TNON', 'BWEN', 'LFWD', 'CODX', 'BANL', 'FOSL', 'YIBO', 'BZFD', 'ALFUW', 'CLIR', 'UONE', 'TAIT', 'MSS', 'DSY', 'NAMMW', 'APWC', 'BFRI', 'HKPD', 'CARM', 'LPRO', 'GOSS', 'FBIO', 'SOHO', 'CGC', 'GTBP', 'NCRA', 'HIVE', 'YYAI', 'ANNAW', 'CLWT', 'EVLVW', 'GANX', 'FUFUW', 'ABP', 'QH', 'HAO', 'MTEKW', 'FERAR', 'RZLV', 'NVFY', 'TNMG', 'HLP', 'QIPT', 'OAKUR', 'SKBL', 'MOBX', 'CHRS', 'BZUN', 'CRGO', 'PTLE', 'GNSS', 'LGCB', 'SMXT', 'SXTC', 'DAICW', 'ILLR', 'CNTX', 'NVNI', 'EVTV', 'MYNZ', 'GERN', 'IRD', 'IPW', 'ADVB', 'VIVS', 'BDTX', 'PRZO', 'IPODW', 'KLRS', 'USIO', 'DRDBW', 'CPSH', 'MTVA', 'HUMAW', 'NTRBW', 'USEG', 'VERU', 'TSHA', 'CUBWW', 'OPTX', 'EOSEW', 'MOBBW', 'MYSZ', 'ILAG', 'CIFRW', 'CAN', 'PGEN', 'AACBR', 'ELPW', 'HLVX', 'LESL', 'ARTV', 'TAVIR', 'ICCM', 'ZCMD', 'TALK', 'TPIC', 'HWH', 'HXHX', 'MGRX', 'SVRA', 'YXT', 'BTBD', 'AISPW', 'SYTAW', 'PSTV', 'BMEA', 'IOBT', 'EDUC', 'YHNAR', 'PN', 'PHH', 'BEEM', 'PACB', 'AYTU', 'APYX', 'GRWG', 'HUMA', 'CLNE', 'SCYX', 'AIRJW', 'BON', 'QRHC', 'PDSB', 'HPKEW', 'RAAQW', 'SGBX', 'MLGO', 'GOVX', 'RMSG', 'SLDP', 'FGMCR', 'NTWOW', 'ESLA', 'TNYA', 'RLMD', 'RVSN', 'BIYA', 'FAT', 'ZEO', 'FGL', 'GLBS', 'NOEMR', 'PAVM', 'FATBB', 'NLSP', 'AQMS', 'GSIW', 'IPHA', 'ACXP', 'ZVSA', 'RVMDW', 'LIDR', 'LXEH', 'SLDPW', 'SHMDW', 'FOXXW', 'DVLT', 'VSEE', 'ELAB', 'RDAGW', 'PPSI', 'WETH', 'PT', 'RDGT', 'WLACW', 'BCTX', 'ALDFW', 'RNWWW', 'AGRI', 'MKZR', 'SAGT', 'YHGJ', 'VSME', 'FRSX', 'MEGL', 'PWM', 'ANL', 'MRSN', 'XTKG', 'RXT', 'EVGN', 'ORKT', 'DYAI', 'RENB', 'SFWL', 'IROHR', 'PODC', 'CVGI', 'SJ', 'NVVE', 'MYPS', 'NIPG', 'ADN', 'ABSI', 'AIRE', 'JZXN', 'XTLB', 'APRE', 'CONI', 'TACOW', 'KAVL', 'CETY', 'IPSC', 'MEIP', 'DHAI', 'GDHG', 'NAOV', 'QSEAR', 'MOGO', 'POAI', 'KWMWW', 'PMAX', 'WENNW', 'DMAAR', 'MGX', 'IOVA', 'SATLW', 'XFOR', 'ABLV', 'UTSI', 'PET', 'MNTS', 'ZURA', 'ORGN', 'BYSI', 'OMEX', 'WOK', 'YYGH', 'BTOC', 'PRTS', 'XAGE', 'JCSE', 'KIRK', 'EGHT', 'BLDEW', 'FTHM', 'CLGN', 'DXLG', 'LOBO', 'ENGNW', 'AMPGW', 'PSHG', 'AAME', 'OSRH', 'STTK', 'IMNN', 'BFRGW', 'TRAW', 'GWAV', 'ZDAI', 'GTEC', 'NDLS', 'SNDL', 'SLXN', 'TDTH', 'ACIU', 'ZKIN', 'ATXG', 'TDACW', 'MCHX', 'POLA', 'LPSN', 'LRE', 'SMTK', 'ADAP', 'AGH', 'PPBT', 'MNDR', 'AIXI', 'NXPL', 'FAMI', 'SERA', 'MDXH', 'IRWD', 'ZENV', 'THAR', 'TYGO', 'HURA', 'PSNYW', 'OXBRW', 'VS', 'SNGX', 'JZ', 'GEG', 'GIPRW', 'PNBK', 'DCGO', 'VRAX', 'VANI', 'LVTX', 'GIFT', 'INMB', 'YQ', 'SUNE', 'ECDA', 'SUUN', 'OPI', 'UFG', 'SKIN', 'SEATW', 'LMFA', 'ENTX', 'GUTS', 'COSM', 'ICG', 'WLDS', 'PLRX', 'CTXR', 'IINN', 'KLXE', 'OTRK', 'WFF', 'GRI', 'GPATW', 'MKDW', 'ARBK', 'LPTX', 'HOVR', 'MLACR', 'WATT', 'CGCTW', 'STAK', 'XHG', 'GROW', 'CISO', 'DTCK', 'RANI', 'CAPS', 'PMN', 'ATOS', 'VOR', 'SXTP', 'PETZ', 'XRTX', 'ATPC', 'OP', 'CCG', 'HVIIR', 'TIRX', 'RMCO', 'NPACW', 'JAGX', 'KTTA', 'TMCWW', 'SGLY', 'TRIB', 'ALVOW', 'KNDI', 'SPKLW', 'MDIA', 'SCNX', 'XELB', 'DLTH', 'ZOOZ', 'SANW', 'AEI', 'CRNT', 'ICON', 'CRBU', 'AMOD', 'NIOBW', 'ABOS', 'TOP', 'ASPSW', 'IXHL', 'AKAN', 'FBLG', 'FATE', 'CSAI', 'NERV', 'INTZ', 'JUNS', 'RDZN', 'BZAIW', 'SBFM', 'NXXT', 'OPK', 'GREE', 'ADVM', 'CETX', 'SOPA', 'WINT', 'INTS', 'IVP', 'FARM', 'VRA', 'HRTX', 'CTMX', 'KLTO', 'LOT', 'PRLD', 'VEEAW', 'QSIAW', 'CMMB', 'COCP', 'HTCR', 'XAIR', 'PRPL', 'LEXX', 'ADIL', 'TUSK', 'SVRE', 'HONDW', 'AUTL', 'DARE', 'CELU', 'RAY', 'BFRG', 'OESX', 'ENGS', 'PLRZ', 'JBDI', 'SOWG', 'SILO', 'LUCY', 'FFAI', 'LVRO', 'BEAT', 'EU', 'UCL', 'HOWL', 'BEAGR', 'CGTX', 'TXMD', 'OMH', 'DTSS', 'REFR', 'NETDW', 'TROO', 'ADGM', 'BITF', 'AENTW', 'RTACW', 'HPAIW', 'DYCQR', 'UYSCR', 'MRM', 'XWEL', 'CNDT', 'GPRO', 'BIAFW', 'GFAIW', 'MIGI', 'MREO', 'DFLI', 'HAIN', 'CAPT', 'SPHL', 'LGO', 'USGOW', 'OPEN', 'LIXTW', 'NITO', 'RNTX', 'CGEN', 'MNY', 'INHD', 'WAI', 'ELUT', 'HCAI', 'LWLG', 'LCID', 'ATMVR', 'AKTX', 'OCTO', 'PRFX', 'OABIW', 'QSI', 'RSLS', 'POWW', 'INLF', 'DWSN', 'WCT', 'ASPSZ', 'TLSIW', 'AREB', 'GP', 'PFAI', 'MULN', 'KLTR', 'CCIXW', 'CLLS', 'CENN', 'CCLD', 'CGTL', 'RELI', 'IMUX', 'SDHIR', 'ACRS', 'FGI', 'VUZI', 'HBIO', 'AMPG', 'WORX', 'ENVB', 'SGRP', 'PRSO', 'EXFY', 'ATMCR', 'BOXL', 'GURE', 'TLPH', 'HOVRW', 'HUBC', 'GGR', 'CABA', 'LUCD', 'ORMP', 'UGRO', 'GORV', 'MNTK', 'GTIM', 'CNET', 'LVO', 'TGL', 'IVF', 'YAAS', 'ADD', 'TOUR', 'SEED', 'WALD', 'IFRX', 'BCG', 'NWGL', 'INEO', 'JYD', 'CNCKW', 'CHEK', 'PLMKW', 'ABAT', 'NNBR', 'OKUR', 'RDHL', 'OPAL', 'EVGOW', 'NIVF', 'ASBP', 'YTRA', 'IKT', 'NMRA', 'PASG', 'NXTC', 'SSKN', 'MBRX', 'ICU', 'BGLWW', 'LPAAW', 'PDYNW', 'RR', 'OTLK', 'LNZA', 'VACHW', 'ZTEK', 'CHARR', 'EHGO', 'SMSI', 'RCON', 'MGNX', 'CNFR', 'RAYA', 'PSIG', 'CTRM', 'IVVD', 'PMTRW', 'SELX', 'LEXXW', 'BENF', 'DSWL', 'CHR', 'GRYP', 'GFAI', 'KXIN', 'ASRT', 'OXBR', 'CNTB', 'EEIQ', 'EPIX', 'VIVK', 'LIQT', 'LGVN', 'DGXX', 'ACET', 'CTSO', 'SYBX', 'GALT', 'RIME', 'MXCT', 'EPOW', 'NEOVW', 'PSNY', 'IPDN', 'ONDS', 'STRR', 'BTOG', 'CMBM', 'CCIRW', 'BCTXZ', 'AREC', 'SHIM', 'CARV', 'BTAI', 'LOOP', 'CURR', 'PRQR', 'BLNK', 'ASTI', 'EFOI', 'COLAR', 'CASK', 'NHICW', 'VRAR', 'FMSTW', 'IMMX', 'ANNX', 'VTGN', 'RMCF', 'GDTC', 'BOWNR', 'RMTI', 'BEATW', 'ATER', 'GELS', 'TBLAW', 'REE', 'TLSA', 'SKYQ', 'NAMI', 'YHC', 'CCCC', 'ADV', 'TOMZ', 'DGLY', 'TANH', 'BHAT', 'AACG', 'LUNG', 'HITI', 'PMEC', 'SLNH', 'SUGP', 'ONFO', 'MIST', 'BSLK', 'ANY', 'HOUR', 'INAB', 'FACTW', 'BACQR', 'INO', 'ZBAI', 'NCEW', 'BLIN', 'MSAI', 'ATAI', 'CRON', 'INCR', 'PHIO', 'CLSD', 'UOKA', 'WYHG', 'ADSEW', 'GIPR', 'LHSW', 'RVPH', 'ANSCW', 'TBMCR', 'WHWK', 'ALBT', 'CUE', 'VRME', 'FTRK', 'NAUT', 'EPWK', 'FAAS', 'JXG', 'ANTE', 'LBGJ', 'XCH', 'CNSP', 'MURA', 'ISRLW', 'CRMLW', 'GMGI', 'KOPN', 'CRESW', 'TACHW', 'HSPOR', 'NAAS', 'TURB', 'SDA', 'HIHO', 'YJ', 'PC', 'AIHS', 'GTENW', 'PAVS', 'BAYAR', 'CDT', 'FNGR', 'OLPX', 'GBIO', 'SPWRW', 'MACIW', 'ZJYL', 'MCVT', 'SDST', 'VEEA', 'LCCCR', 'MSPR', 'REKR', 'SCNI', 'GAME', 'VCICW', 'RNXT', 'HIT', 'STAI', 'ARBE', 'HUIZ', 'SVC', 'TCRX', 'QMMM', 'QNCX', 'BRTX', 'DAAQW', 'MIRA', 'ODVWZ', 'CLPS', 'DAVEW', 'DUO', 'USEA', 'FBYDW', 'EDAP', 'MAYAR', 'DEVS', 'IRIX', 'FSHPR', 'VERI', 'BCAB', 'SGD', 'MVIS', 'PEPG', 'MHUA', 'VEEE', 'KITT', 'SNTI', 'ARAY', 'ELEV', 'GRABW', 'LGCL', 'IMMP', 'ALLO', 'BJDX', 'MTC', 'GEVO', 'PMVP', 'UONEK', 'JFU', 'IFBD', 'TELA', 'LSTA', 'CRGOW', 'RETO', 'FORA', 'MBIO', 'FTFT', 'HGBL', 'EZGO', 'SYPR', 'OABI', 'MRKR', 'OCGN', 'TVGN', 'GSHRW', 'FPAY', 'PELIR', 'ACHV', 'STEC', 'ASTLW', 'OXSQ', 'ELBM', 'CCCXW', 'BCDA', 'BRLT', 'XLO', 'BRNS', 'COOT', 'CALC', 'ZNTL', 'RSVRW', 'AFRIW', 'CDROW', 'SKYX', 'JWEL', 'IPA', 'EURKR', 'HUDI', 'STFS', 'SNAL', 'OFAL', 'ENSC', 'DATSW', 'DYNXW', 'MOVE', 'SGMO', 'FLNT', 'DXST', 'ERNA', 'SNTG', 'GLE', 'HERZ', 'ATLN', 'SEGG', 'HTCO', 'LOKVW', 'DATS', 'SFHG', 'PMCB', 'HOOK', 'NIXX', 'EM', 'ABLLW', 'CLSKW', 'OVID', 'RCT', 'GLXG', 'ERAS', 'FVNNR', 'RMBL', 'TEAD', 'WRAP', 'IQ', 'CLYM', 'BAOS', 'DRTSW', 'CRAQR', 'MWYN', 'CREG', 'BGFV', 'GNPX', 'VFF', 'ZYXI', 'ZBAO', 'IMRN', 'RZLVW', 'EDBL', 'DRRX', 'AQB', 'CERS', 'CXAI', 'VERO', 'BAER', 'INTJ', 'PCSA', 'CSTE', 'SBCWW', 'LAB', 'CHACR', 'GV', 'XPON', 'SEER', 'VYNE', 'ATIIW', 'VGASW', 'CBAT', 'MBOT', 'BDRX', 'HYPR', 'TSBX', 'BKHAR', 'ARQQW', 'IPM', 'CMND', 'CRDL', 'CCCMW', 'ITRM', 'NOTV', 'ATHA', 'OGI', 'NRSNW', 'VCIG', 'RANGR', 'AFJKR', 'HOTH', 'XTIA', 'HMR', 'BNRG', 'IINNW', 'BIAF', 'SAVA', 'PYXS', 'SHOT', 'DLPN', 'IMTE', 'SVCCW', 'SZZLR', 'OLB', 'SIMAW', 'LIMNW', 'AHG', 'AMIX', 'CDTG', 'AGAE', 'IZM', 'IVDA', 'DIBS', 'TVACW', 'AWRE', 'IMAB', 'TELO', 'VRCA', 'FTEL', 'CNEY', 'NWTN', 'SWAGW', 'EDTK', 'AERT', 'SWAG', 'GXAI', 'BKYI', 'EVAX', 'ALTO', 'UHGWW', 'SKK', 'TVAIR', 'CJET', 'STRO', 'CASI', 'APLT', 'FLGC', 'SAFX', 'ARBEW', 'AEVAW', 'ECX', 'NSPR', 'ISPC', 'ACRV', 'ATNF', 'RDI', 'SDOT', 'GCMGW', 'FLUX', 'USARW', 'GRRRW', 'ESPR', 'BDSX', 'RLYB', 'GRNQ', 'AEMD', 'SISI', 'CAPNR', 'IGMS', 'CAMP', 'ALLR', 'HTOO', 'ORIS', 'XHLD', 'TBH', 'NXGL', 'UROY', 'SONM', 'EQ', 'CRIS', 'COCH', 'DRCT', 'UPLD', 'NMTC', 'PLUG', 'LSH', 'LPBBW', 'APM', 'MDAIW', 'EJH', 'AACIW', 'MBAVW', 'IKNA', 'NEHC', 'MASK', 'CCTG', 'OACCW', 'ENTO', 'CDLX', 'FEMY', 'GLMD', 'ADAG', 'BLNE', 'FEBO', 'NWTG', 'PLBY']
symbols = list(set(symbols))

API_RATE_LIMIT_MIN = 200
def run_tracker(symbols=symbols, batch_size=API_RATE_LIMIT_MIN, sleep_after_batch=60):
    print(f'Tracking {len(symbols)} symbols in batches of {batch_size}…')
    results = []
    calls = 0

    next_sym = []
    for sym in symbols:
        calls += 1
        # Throttle to respect the rate limit
        if calls > 0 and calls % batch_size == 0:
            print(f'[!] Reached {calls} API calls, sleeping {sleep_after_batch}s…')
            time.sleep(sleep_after_batch)

        res = analyze_stock(sym)
        if not res:
            continue
        next_sym.append(sym) # Remove securities that are not day-tradable, etc.

        score = 0
        score += 0.1 * min(res['percent_change'] / 10, 1) # momentum
        score += 0.1 * (1 - res['RSI3'] / 100) # RSI
        score += 0.1 * (1 if res['first_ema_cross'] else 0) # EMA cross?
        score += 0.13 * (1 if res['last_price'] > res['VWAP'] else 0) # price above VWAP?
        score += 0.1 * res['breakout_run'] # breakout 
        score += 0.1 * (1 if res['rsi_cross50'] else 0) # RSI cross?
        score += 0.07 * res['dip_score'] # dip?
        score += 0.3 * min(1, res['volume'] / 500_000) # volume - I weighted this strongly because I wanted to take short-term positions and trade securities that were very active
        score -= 0.05 * res['runup_penalty'] # if a stock has already run-up, caution against buying it

        score = max(0, score) # already clipped to [0, 1]
        res['score'] = round(score, 3)

        results.append(res)

    if not results:
        print('No symbols passed filters this run.')
        return

    df = pd.DataFrame(results).sort_values('score', ascending=False)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df.insert(0, 'timestamp', ts)

    print(f'\n[!!] Update at {datetime.now().strftime('%H:%M:%S')}:')
    print(df)

    append_to_google_sheet(
        df,
        spreadsheet_id='1R4mPwcXqK8dUfj_CwKR1eknAf1saIRpLZzZtPooV4bw',
        range_name='Sheet1!A1'
    )
    return next_sym # passed into the next run of the function

# Your credentials should be in `tradingkey.json` in the same directory as this file
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

def append_to_google_sheet(df, spreadsheet_id, range_name, creds_path='tradingkey.json'):
    creds = Credentials.from_service_account_file(creds_path, scopes=[
        'https://www.googleapis.com/auth/spreadsheets'
    ])
    service = build('sheets', 'v4', credentials=creds)
    df_clean = df.fillna('').astype(str)
    values = [df_clean.columns.tolist()] + df_clean.values.tolist()

    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption='RAW',
        body={'values': values}
    ).execute()


if __name__ == '__main__':
    # symbols = get_penny_stocks()
    # print(symbols)

    while True:
        symbols = run_tracker()
        time.sleep(600)

