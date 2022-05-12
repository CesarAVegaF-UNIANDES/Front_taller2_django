from django.shortcuts import render

from app.main import getMyItems, getRecommendBPR, getRecommendWARP


def loginView(request):
    return render(request, '../templates/recomendaciones.html')


def recomendacionesView(request):
    userIDGlobal = int(request.GET["user_id"])
    recommended_user = getMyItems(userIDGlobal)
    prediccionesBPR, details_BPR, lat_center_BPR, long_center_BPR = getRecommendBPR(
        userIDGlobal)
    prediccionesWARP, details_WARP, lat_center_WARP, long_center_WARP = getRecommendWARP(
        userIDGlobal)

    f = open('templates/recomendaciones.html', 'r')
    result = ""
    for line in f.readlines():
        result = result + line
    f.close()

    result = result.replace("mis_calificaciones", recommended_user.to_html())
    result = result.replace("mis_recomendados_BRP", prediccionesBPR.to_html())
    result = result.replace("mis_recomendados_WARP",
                            prediccionesWARP.to_html())
    result = result.replace("marksBRP_center_Lat", str(lat_center_BPR))
    result = result.replace("marksBRP_center_Long", str(long_center_BPR))
    result = result.replace("marksWARP_center_Lat", str(lat_center_WARP))
    result = result.replace("marksWARP_center_Long", str(long_center_WARP))
    result = result.replace("marksBRP", str(details_BPR.to_numpy().tolist()))
    result = result.replace("marksWARP", str(details_WARP.to_numpy().tolist()))

    result = result.replace("right", "center")
    result = result.replace("dataframe", "default")
    result = result.replace("<th></th>", "<th scope=\"row\">item_id</th>")
    result = result.replace("<tr>", "<tr valign=\"center\" align=\"center\">")

    f = open('templates/recomendaciones_modified.html', 'w')
    f.write(result)
    f.close()

    return render(request, '../templates/recomendaciones_modified.html')
