import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import plotly.express as px
import plotly.graph_objects as go

import matplotlib
matplotlib.use('Agg')  # Compatible with Streamlit

# ----------------------
# Define the MNIST model
# ----------------------
class ConvolutionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

@st.cache_resource
def load_model():
    model_path = "mnist_model.pth"
    model = ConvolutionNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def register_hooks(model, activations_dict):
    def get_activation(name):
        def hook(model, input, output):
            activations_dict[name] = output.detach()
        return hook
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.fc1.register_forward_hook(get_activation('fc1'))
    model.fc2.register_forward_hook(get_activation('fc2'))

# CSS to limit main block width to 50%
st.markdown(
    """
    <style>
    .main > div {
        max-width: 50%;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if 'activations' not in st.session_state:
    st.session_state.activations = {}
if 'pred' not in st.session_state:
    st.session_state.pred = None
if 'probs' not in st.session_state:
    st.session_state.probs = None

st.title("Handwritten Digit Inference MNIST")

# Create two columns: one for the canvas and one for the prediction
col1, col2 = st.columns(2)

with col1:
    st.write("Draw a digit (0-9) below, then click 'PREDICT'.")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Clear old results
        st.session_state.activations.clear()

        # Register hooks on fresh dictionary
        register_hooks(model, st.session_state.activations)

        img = Image.fromarray((canvas_result.image_data).astype('uint8'), mode="RGBA")
        img = img.convert('L')
        img = img.resize((28, 28))
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.session_state.pred = pred
        st.session_state.probs = probs.numpy().squeeze()

# Display predicted digit in the second column (if available)
with col2:
    if st.session_state.pred is not None:
        st.markdown(
            f"<h1 style='font-size:60px; font-family:'Roboto'; font-weight:bold; text-align:center;'>Prediction: {st.session_state.pred}</h1>",
            unsafe_allow_html=True
        )

# Only proceed if we have predictions in session state
if st.session_state.pred is not None and st.session_state.probs is not None:
    st.write(f"**Predicted Digit:** {st.session_state.pred}")

    # Probability bar chart
    prob_array = st.session_state.probs
    prob_df = {
        'Digit': np.arange(10),
        'Probability': prob_array
    }
    fig = px.bar(prob_df, x='Digit', y='Probability', title='Prediction Probabilities')
    st.plotly_chart(fig)

    # Retrieve activations
    activations = st.session_state.activations

    # Show 2D feature maps for conv layers
    st.write("## Conv Layers 2D Feature Maps")
    if 'conv1' in activations:
        conv1_act = activations['conv1'].squeeze(0)
        num_filters = conv1_act.shape[0]
        fig, axs = plt.subplots(1, num_filters, figsize=(num_filters*2.5, 2.5))
        if num_filters == 1:
            axs = [axs]
        for i in range(num_filters):
            axs[i].imshow(conv1_act[i].numpy(), cmap='viridis')
            axs[i].axis('off')
            axs[i].set_title(f'F{i}')
        st.pyplot(fig)

    if 'conv2' in activations:
        conv2_act = activations['conv2'].squeeze(0)
        num_filters = conv2_act.shape[0]
        cols = 4
        rows = (num_filters // cols) + (num_filters % cols > 0)
        fig, axs = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
        axs = axs.flatten()
        for i in range(num_filters):
            axs[i].imshow(conv2_act[i].numpy(), cmap='viridis')
            axs[i].axis('off')
            axs[i].set_title(f'F{i}')
        for j in range(i+1, len(axs)):
            axs[j].axis('off')
        st.pyplot(fig)

    # Separate 3D Convolution Graphs Side by Side
    st.write("## 3D Visualization of Convolution Layers")

    if 'conv1' in activations and 'conv2' in activations:
        conv1_data = activations['conv1'].squeeze(0).numpy()  # shape (6,13,13)
        c1_nf, c1_h, c1_w = conv1_data.shape
        conv2_data = activations['conv2'].squeeze(0).numpy()  # shape (16,5,5)
        c2_nf, c2_h, c2_w = conv2_data.shape

        col1_3d, col2_3d = st.columns(2)

        # Slider for Conv1
        selected_filter_conv1 = col1_3d.slider(
            "Conv1 Filter Index (0=Off):", 
            min_value=0, 
            max_value=c1_nf, 
            value=0
        )
        # Slider for Conv2
        selected_filter_conv2 = col2_3d.slider(
            "Conv2 Filter Index (0=Off):", 
            min_value=0, 
            max_value=c2_nf, 
            value=0
        )

        # Handle Conv1 plotting
        x_c1, y_c1, z_c1 = np.meshgrid(np.arange(c1_w), np.arange(c1_h), np.arange(c1_nf))
        x_c1_flat, y_c1_flat, z_c1_flat = x_c1.flatten(), y_c1.flatten(), z_c1.flatten()
        c1_values = conv1_data.flatten()
        c1_min, c1_max = c1_values.min(), c1_values.max()
        c1_norm = (c1_values - c1_min) / (c1_max - c1_min + 1e-9)

        fig_conv1_3d = go.Figure()
        if selected_filter_conv1 == 0:
            # All filters
            fig_conv1_3d.add_trace(go.Scatter3d(
                x=x_c1_flat, y=y_c1_flat, z=z_c1_flat,
                mode='markers',
                marker=dict(
                    size=3,
                    color=c1_norm,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Conv1 Activation')
                ),
                name='Conv1'
            ))
        else:
            filt_idx = selected_filter_conv1 - 1
            mask = z_c1_flat == filt_idx
            fig_conv1_3d.add_trace(go.Scatter3d(
                x=x_c1_flat[mask], y=y_c1_flat[mask], z=z_c1_flat[mask],
                mode='markers',
                marker=dict(
                    size=3,
                    color=c1_norm[mask],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=f'Conv1_F{filt_idx}')
                ),
                name=f'Conv1_F{filt_idx}'
            ))

        fig_conv1_3d.update_layout(
            scene=dict(
                xaxis_title='X Dim',
                yaxis_title='Y Dim',
                zaxis_title='Filter Index',
                aspectmode='cube'
            ),
            title='Conv1 Activations' + (" - All" if selected_filter_conv1==0 else "")
        )
        col1_3d.plotly_chart(fig_conv1_3d)

        # Handle Conv2 plotting
        x_c2, y_c2, z_c2 = np.meshgrid(np.arange(c2_w), np.arange(c2_h), np.arange(c2_nf))
        x_c2_flat, y_c2_flat, z_c2_flat = x_c2.flatten(), y_c2.flatten(), z_c2.flatten()
        c2_values = conv2_data.flatten()
        c2_min, c2_max = c2_values.min(), c2_values.max()
        c2_norm = (c2_values - c2_min) / (c2_max - c2_min + 1e-9)

        fig_conv2_3d = go.Figure()
        if selected_filter_conv2 == 0:
            # All filters
            fig_conv2_3d.add_trace(go.Scatter3d(
                x=x_c2_flat, y=y_c2_flat, z=z_c2_flat,
                mode='markers',
                marker=dict(
                    size=3,
                    color=c2_norm,
                    colorscale='Inferno',
                    showscale=True,
                    colorbar=dict(title='Conv2 Activation')
                ),
                name='Conv2'
            ))
        else:
            filt_idx = selected_filter_conv2 - 1
            mask = z_c2_flat == filt_idx
            fig_conv2_3d.add_trace(go.Scatter3d(
                x=x_c2_flat[mask], y=y_c2_flat[mask], z=z_c2_flat[mask],
                mode='markers',
                marker=dict(
                    size=3,
                    color=c2_norm[mask],
                    colorscale='Inferno',
                    showscale=True,
                    colorbar=dict(title=f'Conv2_F{filt_idx}')
                ),
                name=f'Conv2_F{filt_idx}'
            ))

        fig_conv2_3d.update_layout(
            scene=dict(
                xaxis_title='X Dim',
                yaxis_title='Y Dim',
                zaxis_title='Filter Index',
                aspectmode='cube'
            ),
            title='Conv2 Activations' + (" - All" if selected_filter_conv2==0 else "")
        )
        col2_3d.plotly_chart(fig_conv2_3d)

    # FC Layers 3D Visualization
    if 'fc1' in activations and 'fc2' in activations:
        st.write("## 3D Visualization of Fully Connected Layers")
        fc1_act = activations['fc1'].squeeze(0).numpy()
        fc2_act = activations['fc2'].squeeze(0).numpy()

        # Reshape as a pseudo 3D just for visualization
        fc1_grid = fc1_act.reshape(8,5,3) 
        fc2_grid = fc2_act.reshape(7,6,2)

        # Flatten fc1
        fc1_x_dim = fc1_grid.shape[2]
        fc1_y_dim = fc1_grid.shape[1]
        fc1_z_dim = fc1_grid.shape[0]
        x_coords_fc1, y_coords_fc1, z_coords_fc1 = np.meshgrid(
            np.arange(fc1_x_dim),
            np.arange(fc1_y_dim),
            np.arange(fc1_z_dim)
        )
        fc1_x_flat = x_coords_fc1.flatten()
        fc1_y_flat = y_coords_fc1.flatten()
        fc1_z_flat = z_coords_fc1.flatten()
        fc1_values = fc1_grid.flatten()
        val_min_fc1, val_max_fc1 = fc1_values.min(), fc1_values.max()
        fc1_val_norm = (fc1_values - val_min_fc1) / (val_max_fc1 - val_min_fc1 + 1e-9)

        # Flatten fc2
        fc2_x_dim = fc2_grid.shape[2]
        fc2_y_dim = fc2_grid.shape[1]
        fc2_z_dim = fc2_grid.shape[0]
        x_coords_fc2, y_coords_fc2, z_coords_fc2 = np.meshgrid(
            np.arange(fc2_x_dim),
            np.arange(fc2_y_dim),
            np.arange(fc2_z_dim)
        )
        fc2_x_flat = x_coords_fc2.flatten() + fc1_x_dim + 1  # offset for clarity
        fc2_y_flat = y_coords_fc2.flatten()
        fc2_z_flat = z_coords_fc2.flatten()
        fc2_values = fc2_grid.flatten()
        val_min_fc2, val_max_fc2 = fc2_values.min(), fc2_values.max()
        fc2_val_norm = (fc2_values - val_min_fc2) / (val_max_fc2 - val_min_fc2 + 1e-9)

        fig_fc = go.Figure()

        # FC1 trace
        fig_fc.add_trace(go.Scatter3d(
            x=fc1_x_flat,
            y=fc1_y_flat,
            z=fc1_z_flat,
            mode='markers',
            marker=dict(
                size=5,
                color=fc1_val_norm,
                colorscale='Inferno',
                showscale=True,
                colorbar=dict(title='FC1 Activation')
            ),
            name='FC1'
        ))

        # FC2 trace
        fig_fc.add_trace(go.Scatter3d(
            x=fc2_x_flat,
            y=fc2_y_flat,
            z=fc2_z_flat,
            mode='markers',
            marker=dict(
                size=5,
                color=fc2_val_norm,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title='FC2 Activation')
            ),
            name='FC2'
        ))

        fig_fc.update_layout(
            scene=dict(
                xaxis_title='X Dim',
                yaxis_title='Y Dim',
                zaxis_title='Z Dim',
                aspectmode='cube'
            ),
            title='FC1 and FC2 Activations in 3D'
        )
        st.plotly_chart(fig_fc)

    # Dynamic network graph
    st.write("## Model Architecture Graph")
    view_type = st.radio("Select View Type:", ["High-Level View", "Detailed Conv View"])

    if view_type == "High-Level View":
        # Simple chain: Input -> Conv1 -> Conv2 -> FC1 -> FC2 -> FC3 -> Output
        layers = ["Input", "Conv1", "Conv2", "FC1", "FC2", "FC3", "Output"]
        x_positions = list(range(len(layers)))
        y_positions = [0]*len(layers)

        fig_arch = go.Figure()

        # Nodes
        fig_arch.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='markers+text',
            text=layers,
            textposition='top center',
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(color='black', width=2)
            )
        ))

        # Edges
        for i in range(len(layers)-1):
            fig_arch.add_annotation(
                x=(x_positions[i] + x_positions[i+1])/2,
                y=0,
                axref='x', ayref='y',
                xref='x', yref='y',
                text='',
                ax=x_positions[i],
                ay=0,
                showarrow=True,
                arrowhead=3,
                arrowwidth=2,
                arrowcolor='black'
            )

        fig_arch.update_layout(
            title="Neural Network Flow (High-Level)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        for i in range(len(layers)-1):
            fig_arch['layout']['annotations'][i]['arrowcolor'] = 'white'

        st.plotly_chart(fig_arch)

    else:
        # Detailed Conv View
        nodes = ["Input"]
        conv1_filters = [f"Conv1_F{i}" for i in range(6)]
        conv2_filters = [f"Conv2_F{i}" for i in range(16)]
        nodes.extend(conv1_filters)
        nodes.extend(conv2_filters)
        nodes.append("FC_Block")
        nodes.append("Output")

        input_x = [0]; input_y = [0]
        conv1_x = [1]*len(conv1_filters)
        conv1_y = np.linspace(-2,2,len(conv1_filters))
        conv2_x = [2]*len(conv2_filters)
        conv2_y = np.linspace(-3,3,len(conv2_filters))
        fc_x = [3]; fc_y = [0]
        out_x = [4]; out_y = [0]

        x_positions = input_x + list(conv1_x) + list(conv2_x) + fc_x + out_x
        y_positions = input_y + list(conv1_y) + list(conv2_y) + fc_y + out_y

        fig_arch_detail = go.Figure()
        fig_arch_detail.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='markers+text',
            text=nodes,
            textposition='top center',
            marker=dict(
                size=12,
                color='lightblue',
                line=dict(color='black', width=1)
            )
        ))

        # Input -> Conv1 filters
        for i in range(len(conv1_filters)):
            fig_arch_detail.add_annotation(
                x=(0+1)/2, y=(0+conv1_y[i])/2,
                ax=0, ay=0,
                xref='x', yref='y',
                axref='x', ayref='y',
                text='',
                showarrow=True,
                arrowhead=3,
                arrowwidth=1,
                arrowcolor='black'
            )

        # Conv1 -> Conv2 (fully connected)
        for y1 in conv1_y:
            for y2 in conv2_y:
                fig_arch_detail.add_annotation(
                    x=(1+2)/2, y=(y1+y2)/2,
                    ax=1, ay=y1,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    text='',
                    showarrow=True,
                    arrowhead=2,
                    arrowwidth=1,
                    arrowcolor='gray'
                )

        # Conv2 -> FC block
        for y2 in conv2_y:
            fig_arch_detail.add_annotation(
                x=(2+3)/2, y=(y2+0)/2,
                ax=2, ay=y2,
                xref='x', yref='y',
                axref='x', ayref='y',
                text='',
                showarrow=True,
                arrowhead=3,
                arrowwidth=1,
                arrowcolor='black'
            )

        # FC block -> Output
        fig_arch_detail.add_annotation(
            x=(3+4)/2, y=(0+0)/2,
            ax=3, ay=0,
            xref='x', yref='y',
            axref='x', ayref='y',
            text='',
            showarrow=True,
            arrowhead=3,
            arrowwidth=2,
            arrowcolor='black'
        )

        fig_arch_detail.update_layout(
            title="Neural Network Flow (Detailed Conv View)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=50, b=0)
        )

        # Update arrow annotations in the detailed view to white
        for ann in fig_arch_detail['layout']['annotations']:
            ann['arrowcolor'] = 'white'
        st.plotly_chart(fig_arch_detail)

st.write("---")
st.write("### Explanation")
st.markdown("""
This app uses a Convolutional Neural Network (CNN) to predict handwritten digits drawn on a canvas. The model is trained on the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9).            

This Convolutional Neural Network (CNN) takes a hand-drawn digit image as input and predicts which digit (0 through 9) it represents. Here’s how it works step-by-step:

- **Convolutional Layers:**
    The image is first passed through two convolutional layers (conv1 and conv2). These layers apply kernels that detect local patterns such as edges or curves in the image. Each filter produces a feature map, highlighting certain aspects of the digit’s shape.

- **Pooling Layers:**
    After each convolution, a max-pooling operation reduces the spatial size of the feature maps. This helps the network focus on the most prominent features and makes the model more robust to small shifts and distortions in the input image.

- **Flattening and Fully Connected Layers:**
    The pooled feature maps are then “flattened” into a one-dimensional vector, which feeds into a series of fully connected layers (fc1, fc2, and fc3). These layers combine the extracted features to form increasingly complex representations, ultimately outputting a set of ten probabilities—one for each possible digit.

- **Output and Prediction:**
    The final output layer produces a probability distribution over the digits 0–9. The model chooses the digit with the highest probability as its prediction.
    In essence, the CNN learns filters that automatically detect important features of handwritten digits.

**Made by Berke Özkır**
""")
