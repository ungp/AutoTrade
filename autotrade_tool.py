import json
import numpy
import openpyxl
import sys

## Provide the scaling number to scale the table of asks or bids
# asks: table of asks or bids
def get_scale(asks):
    scale = 0.0
    for i, ask in enumerate(asks):
        scale = scale + ask[1]*float(ask[0])
    return scale

## Provide a new table of relative asks and cumulated asks
# asks: table of asks
# scale: scaling to apply to get the relative asks
# price: current price
def floatify_undim(asks, scale, price):
    newAsks = []
    cumulatedAsk = 0.0
    for i, ask in enumerate(asks):
        cumulatedAsk = cumulatedAsk + ask[1]*float(ask[0])/scale
        newAsks.append([100*(float(ask[0]) - price)/price, cumulatedAsk])
    return newAsks

## Return a data-prediction pair given an Excel sheet, a line in the sheet, and columns of the currency pair (e.g. USDT-BTC)
# ndata: number of data in the pair
# sheet: Excel sheet
# datacols: pair of columns in the sheet corresponding to the currency pair
# i: line in the sheet
def data_predict(ndata, sheet, datacols, i, timepredict, lengthpredict):
    di = 0
    ordersSeries = []
    price = float(json.loads(sheet[datacols[0] + str(i + (ndata - 1))].value)["last"])
    while di < ndata:
        orderBook = json.loads(sheet[datacols[1] + str(i + di)].value)
        if di == 0:
            scale = 0.5*(get_scale(orderBook["asks"]) + get_scale(orderBook["bids"]))
        orders = floatify_undim(orderBook["asks"], scale, price)
        orders.reverse()
        orders.extend(floatify_undim(orderBook["bids"], scale, price))
        ordersSeries.append(orders)
        #if i == 1 and di == 1:
            #print(numpy.array(allasksbids))
        di = di + 1
    
    di = 0
    predict = 0.0
    nlengthpredict = int(lengthpredict/60)
    while di < nlengthpredict:
        price_pred = float(json.loads(sheet[datacols[0] + str(i + (ndata - 1) + int(timepredict/60) + di)].value)["last"])
        predict = predict + 100*(price_pred - price)/price
        di = di + 1
    predict = predict / nlengthpredict
    return [ordersSeries, predict]

## Progress bar
# count: current position in the progress bar
# total: maximum position in the progress bar
# status: label to display on the right of the progress bar
def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
