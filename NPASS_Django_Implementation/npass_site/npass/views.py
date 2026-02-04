# Setting Djago parameters

import json
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .ml import predict

def home(request):
    return render(request, "npass/dash_home.html")

@csrf_exempt
@require_POST
def predict_view(request):
    try:
        data = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON")
    try:
        result = predict(data)
        return JsonResponse({"prediction": result})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
