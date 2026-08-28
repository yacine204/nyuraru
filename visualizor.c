#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080

typedef struct {
    int n_nodes;
    char activation[64];
    float *nodes;
    float *weight;
    float *bias;
} Layer;

typedef struct {
    Layer *input_layer;
    Layer **hidden_layer;
    int n_hidden_layers;
    Layer *outputLayer;
} NN;

static NN *g_nn = NULL;
static RenderTexture2D g_boardCanvas;
static float g_board_start_x;
static float g_board_width;
static int g_screen_width;
static int g_screen_height;

static int g_input_grid_w = 0;
static int g_input_grid_h = 0;

static int g_board_modified = 0;
static float g_last_pixels[784] = {0};


void ui_set_input_grid(int grid_w, int grid_h) {
    g_input_grid_w = grid_w;
    g_input_grid_h = grid_h;
}

int ui_board_modified(void) {
    return g_board_modified;
}



void InitInputLayer(NN *neural_network, char *activation, int n_nodes) {
    neural_network->input_layer = malloc(sizeof(Layer));
    if (!neural_network->input_layer) {
        printf("error allocating memory for neural_network->input_layer\n");
        return;
    }
    neural_network->input_layer->n_nodes = n_nodes;
    strncpy(neural_network->input_layer->activation, activation, 63);
    neural_network->input_layer->activation[63] = '\0';
    
    neural_network->input_layer->bias = calloc(n_nodes, sizeof(float));
    neural_network->input_layer->nodes = calloc(n_nodes, sizeof(float));
    neural_network->input_layer->weight = calloc(n_nodes, sizeof(float));
}

void InitHiddenLayer(NN *neural_network, char **activation, int *n_n_nodes, int n_hidden_layers) {
    neural_network->n_hidden_layers = n_hidden_layers;
    neural_network->hidden_layer = malloc(n_hidden_layers * sizeof(Layer*));
    if (!neural_network->hidden_layer) {
        printf("error allocating memory for neural_network->hidden_layer\n");
        return;
    }
    for (int i = 0; i < n_hidden_layers; i++) {
        neural_network->hidden_layer[i] = malloc(sizeof(Layer));
        neural_network->hidden_layer[i]->n_nodes = n_n_nodes[i];
        strncpy(neural_network->hidden_layer[i]->activation, activation[i], 63);
        neural_network->hidden_layer[i]->activation[63] = '\0';
        
        neural_network->hidden_layer[i]->nodes = calloc(n_n_nodes[i], sizeof(float));
        neural_network->hidden_layer[i]->bias = calloc(n_n_nodes[i], sizeof(float));
        neural_network->hidden_layer[i]->weight = calloc(n_n_nodes[i], sizeof(float));
    }
}

void InitOutLayer(NN *neural_network, char *activation, int n_nodes) {
    neural_network->outputLayer = malloc(sizeof(Layer));
    if (!neural_network->outputLayer) {
        printf("error allocating memory for neural_network->outputLayer\n");
        return;
    }
    neural_network->outputLayer->n_nodes = n_nodes;
    strncpy(neural_network->outputLayer->activation, activation, 63);
    neural_network->outputLayer->activation[63] = '\0';
    
    neural_network->outputLayer->bias = calloc(n_nodes, sizeof(float));
    neural_network->outputLayer->nodes = calloc(n_nodes, sizeof(float));
    neural_network->outputLayer->weight = calloc(n_nodes, sizeof(float));
}

void InitNN(NN *nn, char* i_activation, int i_n_nodes, 
            char **h_activation, int *n_n_nodes, int n_hidden_layers, 
            char *o_activation, int o_n_nodes) {
    InitInputLayer(nn, i_activation, i_n_nodes);
    InitHiddenLayer(nn, h_activation, n_n_nodes, n_hidden_layers);
    InitOutLayer(nn, o_activation, o_n_nodes);
}
static float NodeRadius(float spacing_y) {
    float r = spacing_y * 0.4f;
    if (r > 15.0f) r = 15.0f;
    if (r < 1.5f) r = 1.5f;
    return r;
}

static Color ProbColor(float p) {
    if (p < 0) p = 0;
    if (p > 1) p = 1;
    unsigned char g = (unsigned char)(40 + p * 215);
    unsigned char rb = (unsigned char)(40 * (1.0f - p));
    return (Color){ rb, g, rb, 255 };
}


void DrawNN(NN *nn, int screen_width, int screen_height) {
    float start_x = 100.0f;
    float screen_allowed = 0.6f * (float)screen_width;
    int total_columns = 1 + nn->n_hidden_layers + 1;
    float layer_spacing = screen_allowed / (float)(total_columns - 1);

    int max_nodes = nn->input_layer->n_nodes;
    for (int i = 0; i < nn->n_hidden_layers; i++) {
        if (nn->hidden_layer[i]->n_nodes > max_nodes) max_nodes = nn->hidden_layer[i]->n_nodes;
    }
    if (nn->outputLayer->n_nodes > max_nodes) max_nodes = nn->outputLayer->n_nodes;

    float *prev_layer_y = malloc(max_nodes * sizeof(float));
    int prev_layer_count = 0;
    float prev_layer_x = 0;
    int prev_is_image = 0;

    float current_x = start_x;

    // input layer
    if (g_input_grid_w > 0 && g_input_grid_h > 0) {
        
        float img_size = 220.0f;
        float img_x = start_x - img_size / 2.0f + 15.0f;
        float img_y = (screen_height - img_size) / 2.0f;
        float cell = img_size / (float)g_input_grid_w;

        for (int gy = 0; gy < g_input_grid_h; gy++) {
            for (int gx = 0; gx < g_input_grid_w; gx++) {
                int idx = gy * g_input_grid_w + gx;
                float v = nn->input_layer->nodes[idx];
                if (v < 0) v = 0; if (v > 1) v = 1;
                unsigned char c = (unsigned char)(v * 255);
                DrawRectangle((int)(img_x + gx * cell), (int)(img_y + gy * cell),
                               (int)ceilf(cell), (int)ceilf(cell), (Color){c, c, c, 255});
            }
        }
        DrawRectangleLines((int)img_x, (int)img_y, (int)img_size, (int)img_size, LIGHTGRAY);

       
        prev_layer_y[0] = screen_height / 2.0f;
        prev_layer_count = 1;
        prev_layer_x = img_x + img_size;
        prev_is_image = 1;
    } else {
        float spacing_y = (float)screen_height / (float)(nn->input_layer->n_nodes + 1);
        float radius = NodeRadius(spacing_y);
        for (int i = 0; i < nn->input_layer->n_nodes; i++) {
            prev_layer_y[i] = spacing_y * (float)(i + 1);
            Color c = (nn->input_layer->nodes[i] > 0.0f) ? RED : GREEN;
            DrawCircle((int)start_x, (int)prev_layer_y[i], radius, c);
        }
        prev_layer_count = nn->input_layer->n_nodes;
        prev_layer_x = start_x;
    }

    // hidden layers
    for (int i = 0; i < nn->n_hidden_layers; i++) {
        current_x += layer_spacing;
        float spacing_y = (float)screen_height / (float)(nn->hidden_layer[i]->n_nodes + 1);
        float radius = NodeRadius(spacing_y);

        long line_count = (long)prev_layer_count * nn->hidden_layer[i]->n_nodes;
        int draw_lines = prev_is_image || line_count <= 20000; // skip if it'd be a mess

        if (draw_lines) {
            for (int prev_j = 0; prev_j < prev_layer_count; prev_j++) {
                for (int curr_j = 0; curr_j < nn->hidden_layer[i]->n_nodes; curr_j++) {
                    float current_y = spacing_y * (float)(curr_j + 1);
                    Color lc = (Color){70, 70, 80, prev_is_image ? 180 : 60};
                    DrawLine((int)prev_layer_x, (int)prev_layer_y[prev_j], (int)current_x, (int)current_y, lc);
                }
            }
        }

        for (int j = 0; j < nn->hidden_layer[i]->n_nodes; j++) {
            float current_y = spacing_y * (float)(j + 1);
            Color circle_color = (nn->hidden_layer[i]->nodes[j] > 0.0f) ? RED : GREEN;
            DrawCircle((int)current_x, (int)current_y, radius, circle_color);
            if (radius > 4.0f) DrawCircleLines((int)current_x, (int)current_y, radius, BLACK);
            prev_layer_y[j] = current_y;
        }
        prev_layer_count = nn->hidden_layer[i]->n_nodes;
        prev_layer_x = current_x;
        prev_is_image = 0;
    }

    // output layer
    current_x += layer_spacing;
    float spacing_y = (float)screen_height / (float)(nn->outputLayer->n_nodes + 1);
    float radius = NodeRadius(spacing_y);

    for (int prev_j = 0; prev_j < prev_layer_count; prev_j++) {
        for (int curr_j = 0; curr_j < nn->outputLayer->n_nodes; curr_j++) {
            float current_y = spacing_y * (float)(curr_j + 1);
            DrawLine((int)prev_layer_x, (int)prev_layer_y[prev_j], (int)current_x, (int)current_y, LIGHTGRAY);
        }
    }

    for (int i = 0; i < nn->outputLayer->n_nodes; i++) {
        float current_y = spacing_y * (float)(i + 1);
        float p = nn->outputLayer->nodes[i];  
        Color out_color = ProbColor(p);
        DrawCircle((int)current_x, (int)current_y, radius, out_color);
        DrawCircleLines((int)current_x, (int)current_y, radius, BLACK);
        DrawText(TextFormat("%d", i), (int)current_x + 20, (int)(current_y - 8), 16, RAYWHITE);
    }

    free(prev_layer_y);
}

void DrawBoard(RenderTexture2D canvas, float board_x, float board_y) {
    static Vector2 lastMousePos = { 0 };
    Vector2 mousePos = GetMousePosition();

    Vector2 localMousePos = { mousePos.x - board_x, mousePos.y - board_y };

    if (localMousePos.x >= 0 && localMousePos.x < canvas.texture.width &&
        localMousePos.y >= 0 && localMousePos.y < canvas.texture.height) {
        
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            lastMousePos = localMousePos;
        }

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            BeginTextureMode(canvas);
                DrawLineEx(lastMousePos, localMousePos, 10.0f, WHITE);
                DrawCircleV(localMousePos, 3.0f, WHITE);
            EndTextureMode();
            
            lastMousePos = localMousePos;
            g_board_modified = 1;
        }
    }

    if (IsKeyPressed(KEY_C)) {
        BeginTextureMode(canvas);
            ClearBackground(BLACK);
        EndTextureMode();
        g_board_modified = 1;
    }

    DrawTextureRec(
        canvas.texture, 
        (Rectangle){ 0, 0, (float)canvas.texture.width, (float)-canvas.texture.height }, 
        (Vector2){ board_x, board_y}, 
        WHITE
    );
    
    DrawText("Number Prediction: (Press 'C' to clear)", (int)board_x + 10, (int)board_y + 10, 16, LIGHTGRAY);
}

void ui_reset_board_modified(void) {
    g_board_modified = 0;
}

void FreeNN(NN *nn) {
    if (!nn) return;
    if (nn->input_layer) {
        free(nn->input_layer->nodes);
        free(nn->input_layer->bias);
        free(nn->input_layer->weight);
        free(nn->input_layer);
    }
    if (nn->hidden_layer) {
        for (int i = 0; i < nn->n_hidden_layers; i++) {
            if (nn->hidden_layer[i]) {
                free(nn->hidden_layer[i]->nodes);
                free(nn->hidden_layer[i]->bias);
                free(nn->hidden_layer[i]->weight);
                free(nn->hidden_layer[i]);
            }
        }
        free(nn->hidden_layer);
    }
    if (nn->outputLayer) {
        free(nn->outputLayer->nodes);
        free(nn->outputLayer->bias);
        free(nn->outputLayer->weight);
        free(nn->outputLayer);
    }
    free(nn);
}

void ui_init(int screen_width, int screen_height,
             int i_n_nodes, int *hidden_sizes, int n_hidden,
             int o_n_nodes, const char *o_activation) {
    g_screen_width = screen_width;
    g_screen_height = screen_height;

    g_nn = malloc(sizeof(NN));
    if (!g_nn) { printf("error allocating NN\n"); return; }

    char **h_activations = malloc(n_hidden * sizeof(char*));
    for (int i = 0; i < n_hidden; i++) {
        h_activations[i] = "relu"; 
    }

    InitNN(g_nn, "linear", i_n_nodes, h_activations, hidden_sizes, n_hidden,
           (char *)o_activation, o_n_nodes);
    free(h_activations);

    InitWindow(screen_width, screen_height, "nyuraru");
    SetExitKey(KEY_NULL);
    SetTargetFPS(60);

    g_board_start_x = 100.0f + (0.6f * (float)screen_width) + 40.0f;
    g_board_width = (float)screen_width - g_board_start_x - 50.0f;
    g_boardCanvas = LoadRenderTexture((int)g_board_width, screen_height - 100);

    BeginTextureMode(g_boardCanvas);
        ClearBackground(BLACK);
    EndTextureMode();
}

int ui_should_close(void) {
    int result = WindowShouldClose();
    // printf("ui_should_close() called -> %d\n", result);
    fflush(stdout);
    return result;
}
void ui_frame(void) {
    BeginDrawing();
        ClearBackground(DARKGRAY);
        DrawNN(g_nn, g_screen_width, g_screen_height);
        DrawBoard(g_boardCanvas, g_board_start_x, 20.0f);
    EndDrawing();
}

void ui_set_layer_nodes(int layer_idx, float *values, int n) {
    if (!g_nn) return;
    Layer *target = NULL;
    if (layer_idx == 0) {
        target = g_nn->input_layer;
    } else if (layer_idx == g_nn->n_hidden_layers + 1) {
        target = g_nn->outputLayer;
    } else if (layer_idx >= 1 && layer_idx <= g_nn->n_hidden_layers) {
        target = g_nn->hidden_layer[layer_idx - 1];
    }
    if (!target) return;
    for (int i = 0; i < n && i < target->n_nodes; i++) {
        target->nodes[i] = values[i];
    }
}

void ui_close(void) {
    UnloadRenderTexture(g_boardCanvas);
    CloseWindow();
    FreeNN(g_nn);
    g_nn = NULL;
}


void ui_reset_display(void) {
    if (!g_nn) return;

    memset(g_nn->input_layer->nodes, 0, g_nn->input_layer->n_nodes * sizeof(float));

    for (int i = 0; i < g_nn->n_hidden_layers; i++) {
        memset(g_nn->hidden_layer[i]->nodes, 0, g_nn->hidden_layer[i]->n_nodes * sizeof(float));
    }

    memset(g_nn->outputLayer->nodes, 0, g_nn->outputLayer->n_nodes * sizeof(float));

    ui_clear_board();
}

int ui_predict_pressed(void) {
    return IsKeyPressed(KEY_ENTER);
}

int ui_clear_pressed(void) {
    return IsKeyPressed(KEY_C);
}

void ui_get_board_pixels(int grid_w, int grid_h, float *out) {
    Image img = LoadImageFromTexture(g_boardCanvas.texture);
    ImageFlipVertical(&img);
    ImageResize(&img, grid_w, grid_h);
    ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE);

    unsigned char *px = (unsigned char *)img.data;
    for (int i = 0; i < grid_w * grid_h; i++) {
        out[i] = px[i] / 255.0f;
    }
    UnloadImage(img);
}

void ui_clear_board(void) {
    BeginTextureMode(g_boardCanvas);
        ClearBackground(BLACK);
    EndTextureMode();
}

void ui_draw_prediction_label(int digit, float confidence) {
    const char *txt = TextFormat("Prediction: %d (%.1f%%)", digit, confidence * 100.0f);
    DrawText(txt, (int)g_board_start_x + 10, (int)(g_screen_height - 60), 24, RAYWHITE);
}

void ui_frame_with_prediction(int has_prediction, int digit, float confidence) {
    BeginDrawing();
        ClearBackground(DARKGRAY);
        DrawNN(g_nn, g_screen_width, g_screen_height);
        DrawBoard(g_boardCanvas, g_board_start_x, 20.0f);
        if (has_prediction) {
            ui_draw_prediction_label(digit, confidence);
        }
    EndDrawing();
}

// int main(int argc, char *argv[]) {
//     NN *neural_network = malloc(sizeof(NN));
//     if (!neural_network) return 1;

//     int h_nodes_counts[] = {8, 6, 4};
//     char *h_activations[] = {"relu", "relu", "relu"};
//     int num_hidden_layers = 3;

//     InitNN(neural_network, "sigmoid", 5, h_activations, h_nodes_counts, num_hidden_layers, "softmax", 3);

//     InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "nyuraru");
//     SetTargetFPS(60);

//     float board_start_x = 100.0f + (0.6f * (float)SCREEN_WIDTH) + 40.0f; 
//     float board_width = (float)SCREEN_WIDTH - board_start_x - 50.0f;
//     RenderTexture2D boardCanvas = LoadRenderTexture((int)board_width, SCREEN_HEIGHT-100);

//     BeginTextureMode(boardCanvas);
//         ClearBackground(BLACK);
//     EndTextureMode();

//     while (!WindowShouldClose()) {
//         BeginDrawing();
//             ClearBackground(DARKGRAY);
//             DrawNN(neural_network, SCREEN_WIDTH, SCREEN_HEIGHT);
//             DrawBoard(boardCanvas, board_start_x, 20.0f);
//         EndDrawing();
//     }

//     CloseWindow();
//     FreeNN(neural_network);
//     return 0;
// }
