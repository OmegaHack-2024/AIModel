from typing import Union

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
from io import StringIO
from mlmodel.treadstone import predictConsumption, classifyConsumption
app = FastAPI()

@app.post("/predictions/use")
async def create_upload_file(houseLog: UploadFile = File(...)):
    contents = await houseLog.read()

    string_io = StringIO(classifyConsumption(pd.read_csv(StringIO(contents.decode('utf-8')))).to_csv(index=False))

    # Return the CSV data as a response
    return StreamingResponse(iter([string_io.getvalue()]), media_type="text/csv", headers={'Content-Disposition': 'attachment; filename=useprediction.csv'})

@app.post("/predictions/consumption")
async def create_upload_file(houseLog: UploadFile = File(...)):
    contents = await houseLog.read()

    string_io = StringIO(predictConsumption(pd.read_csv(StringIO(contents.decode('utf-8')))).to_csv(index=False))

    # Return the CSV data as a response
    return StreamingResponse(iter([string_io.getvalue()]), media_type="text/csv", headers={'Content-Disposition': 'attachment; filename=consumptionprediction.csv'})