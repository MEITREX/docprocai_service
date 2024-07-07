import controller
from service.DocProcAiService import DocProcAiService

import uvicorn

if __name__ == "__main__":
    service = DocProcAiService()

    app = controller.create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
