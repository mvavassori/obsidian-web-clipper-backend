from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import openai
from html2text import HTML2Text
from starlette.responses import StreamingResponse
import json

app = FastAPI()

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

origins = [
    "chrome-extension://bpiciejemolpdnbjhjdnikdchjihpgig",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

h = HTML2Text()
h.ignore_links = True
h.ignore_images = True


class Summarize(BaseModel):
    text: str


class HTMLtoText(BaseModel):
    text: str


@app.post("/summarize")
async def create_summary(summarize: Summarize):
    # Convert HTML to plain text
    text = h.handle(summarize.text)

    async def generate_chunks():
        try:
            async for chunk in await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes web pages and formats text in Markdown style.",
                    },
                    {
                        "role": "user",
                        "content": f'Summarize the following text in Markdown, use lists to summarize the most important points: "{text}"',
                    },
                ],
                max_tokens=150,
                stream=True,
            ):
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content is not None:
                    yield json.dumps(
                        {"chunk": content}
                    ) + "\n"  # add a newline after each chunk

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(generate_chunks(), media_type="application/json")

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo-16k",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "You are a helpful assistant that summarizes and formats text in Markdown style.",
    #         },
    #         {
    #             "role": "user",
    #             "content": f'Summarize the following text in Markdown: "{text}"',
    #         },
    #     ],
    #     temperature=0.3,
    #     max_tokens=200,
    # )
    # # Return the summarized text
    # return response["choices"][0]["message"]["content"]


@app.post("/htmltotext")
async def html_to_text(htmltotext: HTMLtoText):
    text = h.handle(htmltotext.text)
    return {"text": text}


@app.post("/list-models")
async def list_models():
    response = openai.Model.list()
    return response
