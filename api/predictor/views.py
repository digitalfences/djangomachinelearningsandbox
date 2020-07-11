from django.shortcuts import render
from .apps import PredictorConfig

from django.http import JsonResponse
from rest_framework.views import APIView

class call_model(APIView):
    def get(self,request):
        if request.method == 'GET':
            # get sound from request
            sound = request.GET.get('sound')
            # vectorize sound
            vector = PredictorConfig.vectorizer.transform([sound])
            #predict based on vector
            prediction = PredictorConfig.regressor.predict(vector)[0]
            #build response
            response = {'dog': prediction}
            #return response
            return JsonResponse(response)

# Create your views here.
