from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
import squarify
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import os
from django.conf import settings
import json

def home(request):
    return render(request, 'diagram_automation_module/home.html')

def result(request, diagram_id=None):
    if diagram_id:
        return render(request, 'diagram_automation_module/result.html', {'diagram_id': diagram_id})
    else:
        return HttpResponse("No diagram ID provided.")

def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file-input'):
        uploaded_file = request.FILES['file-input']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        
        try:
            filename = fs.save('uploaded_file.csv', uploaded_file)
            file_path = fs.path(filename)
            
            # Read the saved CSV file into a DataFrame
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()
            print(f"File uploaded successfully. Columns found: {columns}")  # Debug line
            
            # Store the columns in session
            request.session['csv_columns'] = columns
            request.session['csv_filename'] = filename
            
            return render(request, 'diagram_automation_module/select_axis.html', {
                'columns': columns
            })
        except Exception as e:
            print(f"Error during file upload: {str(e)}")  # Debug line
            return HttpResponse(f"Error processing file: {str(e)}")

    return render(request, 'diagram_automation_module/home.html')

def download_file(request, file_name):
    file_path = os.path.join(settings.STATIC_ROOT, 'diagram_automation_module', file_name)
    try:
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/vnd.ms-excel')
            response['Content-Disposition'] = f'attachment; filename={file_name}'
            return response
    except Exception as e:
        return HttpResponse(f"Error downloading file: {str(e)}")

def generate_graph(request):
    if request.method == 'POST':
        # Retrieve user inputs
        graph_type = request.POST.get('graph-type')
        x_axis = request.POST.get('x-axis')
        y_axis = request.POST.get('y-axis')
        z_axis = request.POST.get('z-axis')

        # Get stored columns from session
        columns = request.session.get('csv_columns', [])
        filename = request.session.get('csv_filename')

        print(f"Available columns: {columns}")  # Debug line
        print(f"Selected axes - X: {x_axis}, Y: {y_axis}, Z: {z_axis}")  # Debug line

        if not filename:
            return HttpResponse("Error: No file has been uploaded")

        # Get the uploaded file
        uploaded_file_path = os.path.join(settings.MEDIA_ROOT, filename)
        if not os.path.exists(uploaded_file_path):
            return HttpResponse("Error: Upload file again, session expired")

        try:
            # Read and validate data
            df = pd.read_csv(uploaded_file_path)
            
            if x_axis not in df.columns:
                return HttpResponse(f"Error: X-axis '{x_axis}' not found in columns: {df.columns.tolist()}")
            if y_axis not in df.columns:
                return HttpResponse(f"Error: Y-axis '{y_axis}' not found in columns: {df.columns.tolist()}")
            if graph_type in ['3d_scatter', '3d_surface'] and (not z_axis or z_axis not in df.columns):
                return HttpResponse(f"Error: Z-axis '{z_axis}' not found in columns: {df.columns.tolist()}")

            # Generate the graph
            plt.figure(figsize=(10, 6))
            
            # Graph generation logic based on type
            if graph_type == 'line':
                plt.plot(df[x_axis], df[y_axis])
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
            elif graph_type == 'bar':
                plt.bar(df[x_axis], df[y_axis])
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
            elif graph_type == 'scatter':
                plt.scatter(df[x_axis], df[y_axis])
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
            elif graph_type == 'pie':
                plt.pie(df[y_axis], labels=df[x_axis], autopct='%1.1f%%')
            elif graph_type == 'box':
                sns.boxplot(x=df[x_axis], y=df[y_axis])
            elif graph_type == 'violin':
                sns.violinplot(x=df[x_axis], y=df[y_axis])
            elif graph_type == 'heatmap':
                sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            elif graph_type == 'area':
                plt.fill_between(df[x_axis], df[y_axis], alpha=0.4)
                plt.plot(df[x_axis], df[y_axis], alpha=0.6)
            elif graph_type == 'polar':
                ax = plt.subplot(111, projection='polar')
                ax.plot(df[x_axis], df[y_axis])
            elif graph_type == '3d_scatter':
                ax = plt.axes(projection='3d')
                ax.scatter3D(df[x_axis], df[y_axis], df[z_axis])
                ax.set_zlabel(z_axis)
            elif graph_type == '3d_surface':
                ax = plt.axes(projection='3d')
                X, Y = np.meshgrid(df[x_axis].unique(), df[y_axis].unique())
                Z = df[z_axis].values.reshape(X.shape)
                ax.plot_surface(X, Y, Z, cmap='viridis')
                ax.set_zlabel(z_axis)
            elif graph_type == 'density':
                sns.kdeplot(data=df, x=x_axis, y=y_axis, cmap="Reds", fill=True)
            elif graph_type == 'histogram':
                plt.hist(df[x_axis], bins=20)
            elif graph_type == 'treemap':
                squarify.plot(sizes=df[y_axis], label=df[x_axis], alpha=0.8)
                plt.axis('off')
            elif graph_type == 'sunburst':
                fig = px.sunburst(df, path=[x_axis, y_axis], values=z_axis)
                static_dir = os.path.join(settings.BASE_DIR, 'static', 'diagram_automation_module')
                os.makedirs(static_dir, exist_ok=True)
                fig.write_html(os.path.join(static_dir, 'sunburst.html'))
                return render(request, 'diagram_automation_module/result.html', {
                    'graph_url': '/static/diagram_automation_module/sunburst.html'
                })
            else:
                return HttpResponse(f"Error: Unsupported graph type: {graph_type}")

            # Add title to the graph
            plt.title(f"{graph_type.capitalize()} Plot")

            # Save the generated graph
            static_dir = os.path.join(settings.BASE_DIR, 'static', 'diagram_automation_module')
            os.makedirs(static_dir, exist_ok=True)
            graph_path = os.path.join(static_dir, 'generated_graph.png')
            plt.savefig(graph_path, bbox_inches='tight', dpi=300)
            plt.close()

            return render(request, 'diagram_automation_module/result.html', {
                'graph_url': '/static/diagram_automation_module/generated_graph.png'
            })

        except Exception as e:
            print(f"Error generating graph: {str(e)}")  # Debug line
            return HttpResponse(f"Error generating graph: {str(e)}")

    return HttpResponse("Invalid request method")