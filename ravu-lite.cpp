// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright (C) 2019 Bin Jin. All Rights Reserved.

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#define DIE(...) {fprintf(stderr, __VA_ARGS__);exit(-1);}
#define PRINT(...) {fprintf(stderr, __VA_ARGS__);}

const double eps = FLT_EPSILON;

typedef std::vector<double> vector;
typedef std::vector<vector> matrix;

const int radius = 3;
const int gradient_radius = std::max(radius - 1, 2);
const int quant_angle = 24;
const int quant_strength = 4;
const int quant_coherence = 3;

const double min_strength[quant_strength - 1] = {0.004, 0.016, 0.05};
const double min_coherence[quant_coherence - 1] = {0.25, 0.5};

const int feature_count = (2 * radius - 1) * (2 * radius - 1);

struct hashkey {
    int angle;
    int strength;
    int coherence;

    hashkey(int angle, int strength, int coherence) : angle(angle), strength(strength), coherence(coherence) { }
};

#define POW3(x) ((x) <= 0 ? 0 : (x) * (x) * (x))
static double bicubic(double x)
{
    return (1.0/6.0) * (      POW3(x + 2)
                        - 4 * POW3(x + 1)
                        + 6 * POW3(x)
                        - 4 * POW3(x - 1));
}

void resample_image(const matrix &image, int target_width, int target_height, matrix *res)
{
    int width = image[0].size(), height = image.size();
    int new_width = width, new_height = height;
    bool sep_x;
    if (height != target_height) {
        new_height = target_height;
        sep_x = true;
    } else if (width != target_width) {
        new_width = target_width;
        sep_x = false;
    } else {
        *res = image;
        return;
    }
    matrix sampled(new_height, vector(new_width));
    for (int x = 0; x < new_height; x++) {
        for (int y = 0; y < new_width; y++) {
            double coord;
            if (sep_x) {
                coord = (double)(x + 0.5) / new_height * height - 0.5;
            } else {
                coord = (double)(y + 0.5) / new_width * width - 0.5;
            }
            int base = (int)floor(coord);
            double fract = coord - base;
            double w[4];
            double wsum = 0.0;
            for (int i = 0; i < 4; i++) {
                w[i] = bicubic(i - 1 - fract);
                wsum += w[i];
            }
            // assert(abs(wsum - 1.0) < eps);
            double sum = 0.0;
            for (int i = 0; i < 4; i++) {
                int sx = x, sy = y;
                if (sep_x) {
                    sx = std::max(0, std::min(height - 1, base - 1 + i));
                } else {
                    sy = std::max(0, std::min(width - 1, base - 1 + i));
                }
                sum += w[i] * image[sx][sy];
            }
            sampled[x][y] = sum / wsum;
        }
    }
    resample_image(sampled, target_width, target_height, res);
}

matrix downscale_bicubic(const matrix &image)
{
    int width = image[0].size(), height = image.size();
    assert(width % 2 == 0 && height % 2 == 0);
    matrix ret;
    resample_image(image, width / 2, height / 2, &ret);
    return ret;
}

matrix upscale_bicubic(const matrix &image)
{
    int width = image[0].size(), height = image.size();
    matrix ret;
    resample_image(image, width * 2, height * 2, &ret);
    return ret;
}

std::string to_string(double x)
{
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10 + 1) << x;
    return oss.str();
}

std::string to_string(int x)
{
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

template<typename T>
std::string to_string(std::vector<T> x)
{
    std::string ret = "[";
    for (int i = 0; i < (int)x.size(); i++) {
        if (i > 0)
            ret += ',';
        ret += to_string(x[i]);
    }
    ret += ']';
    return ret;
}

std::string remove_suffix(const std::string &file_name, const std::string &suffix)
{
    if (file_name.size() > suffix.size() &&
            file_name.substr(file_name.size() - suffix.size()) == suffix) {
        return file_name.substr(0, file_name.size() - suffix.size());
    }
    return file_name;
}

matrix read_image(const std::string &file_name)
{
    // for i in *.jpg; do convert -depth 16 -filter Mitchell -resize 75% "$i" -colorspace gray "$(basename "$i" .jpg).pnm"; done
    // for i in *.png; do convert -depth 16 "$i" -colorspace gray "$(basename "$i" .png).pnm"; done
    FILE* image_file = fopen(file_name.c_str(), "rb");
    if (!image_file) {
        DIE("file '%s' not found\n", file_name.c_str());
    }
    char buf[1024];
    fgets(buf, sizeof(buf), image_file);
    if (strncmp(buf, "P5", 2) != 0) {
        DIE("file '%s' is not a valid PGM file (binary)\n", file_name.c_str());
    }
    fgets(buf, sizeof(buf), image_file);
    int width, height, max_value;
    if (sscanf(buf, "%d%d", &width, &height) != 2) {
        DIE("file '%s' is not a valid PGM file (binary)\n", file_name.c_str());
    }
    fgets(buf, sizeof(buf), image_file);
    if (sscanf(buf, "%d", &max_value) != 1 || max_value != 65535) {
        DIE("file '%s' is not a valid PGM file (binary)\n", file_name.c_str());
    }

    matrix ret(height, vector(width));

    uint16_t line[width];

    for (int i = 0; i < height; i++) {
        if ((int)fread(line, sizeof(*line), width, image_file) != width) {
            DIE("file '%s' is not a valid PGM file (binary)\n", file_name.c_str());
        }
        for (int j = 0; j < width; j++) {
            line[j] = (line[j] & 255) << 8 | line[j] >> 8;
            ret[i][j] = (double)line[j] / max_value;
        }
    }
    fclose(image_file);
    return ret;
}

matrix gaussian;

void prepare_gaussian_2d()
{
    const double sigma = 2;
    const int n = 2 * gradient_radius - 1;
    const double center = gradient_radius - 1;
    double sum = 0;
    gaussian.assign(n, vector(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double dist = (i - center) * (i - center) + (j - center) * (j - center);
            double cost = exp(-dist / (2 * sigma * sigma));
            sum += cost;
            gaussian[i][j] = cost;
        }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            gaussian[i][j] /= sum;
}

std::vector<std::vector<std::vector<std::pair<int,int>>>> samplings;

void prepare_samplings()
{
    const int border = radius - 1;
    std::vector<std::vector<std::pair<int,int>>> ma;
    for (int x = -border; x <= border; x++) {
        ma.emplace_back();
        for (int y = -border; y <= border; y++) {
            ma.back().emplace_back(x, y);
        }
    }
    for (int flip = 0; flip < 2; flip ++) {
        for (int rot = 0; rot < 4; rot++) {
            samplings.push_back(ma);
            for (auto &&row : ma) {
                for (auto &&cell : row) {
                    cell = std::make_pair(cell.second, -cell.first);
                }
            }
        }
        for (auto &&row : ma) {
            for (auto &&cell : row) {
                cell.first = -cell.first;
            }
        }
    }
}

int get_transposed_index(const std::vector<std::vector<std::pair<int,int>>> &ma, int index)
{
    const std::pair<int,int> &cell = ma[radius - 2 + 2 * (index / 2)][radius - 2 + 2 * (index % 2)];
    return (int)(cell.first > 0) * 2 + (int)(cell.second > 0);
}

hashkey get_hashkey(const matrix &patch)
{
    const int n = 2 * radius - 1;
    assert(patch.size() == n && patch[0].size() == n);

    assert(gradient_radius <= radius);
    const int gradient_left = radius - gradient_radius;
    const int gradient_right = n - gradient_left;

    // Eigenvalues and eigenvectors of 2x2 matrices
    // http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
    double a = 0, b = 0, d = 0;
    for (int i = gradient_left; i < gradient_right; i++)
        for (int j = gradient_left; j < gradient_right; j++) {
            double gx, gy;
            if (i == 0)
                gx = patch[i + 1][j] - patch[i][j];
            else if (i == n - 1)
                gx = patch[i][j] - patch[i - 1][j];
            else
                gx = (patch[i + 1][j] - patch[i - 1][j]) / 2.0;
            if (j == 0)
                gy = patch[i][j + 1] - patch[i][j];
            else if (j == n - 1)
                gy = patch[i][j] - patch[i][j - 1];
            else
                gy = (patch[i][j + 1] - patch[i][j - 1]) / 2.0;
            double gw = gaussian[i - gradient_left][j - gradient_left];
            a += gx * gx * gw;
            b += gx * gy * gw;
            d += gy * gy * gw;
        }
    // b == c
    const double T = a + d, D = a * d - b * b;
    const double delta = sqrt(fmax(T * T / 4 - D, 0));
    const double L1 = T / 2 + delta, L2 = T / 2 - delta;
    double V1x, V1y;
    if (fabs(b) > eps) {
        V1x = b;
        V1y = L1 - a;
    } else {
        V1x = 1;
        V1y = 0;
    }

    const double sqrtL1 = sqrt(L1), sqrtL2 = sqrt(L2);

    const double theta = fmod(atan2(V1y, V1x) + M_PI, M_PI);
    const double lambda = sqrtL1;
    const double mu = sqrtL1 + sqrtL2 < eps ? 0 : (sqrtL1 - sqrtL2) / (sqrtL1 + sqrtL2);

    const int angle = (int)floor(theta / (M_PI / quant_angle) + eps);
    assert(0 <= angle && angle < quant_angle);
    const int strength = std::lower_bound(min_strength, min_strength + quant_strength - 1, lambda) - min_strength;
    const int coherence = std::lower_bound(min_coherence, min_coherence + quant_coherence - 1, mu) - min_coherence;

    assert(0 <= strength && strength < quant_strength);
    assert(0 <= coherence && coherence < quant_coherence);
    hashkey ret(angle, strength, coherence);
    return ret;
}

bool gaussian_elimination(matrix &a) {
    int n = a.size();
    assert((int)a[0].size() >= n);

    for (int i = 0; i < n; i++) {
        double aii = a[i][i];
        if (fabs(aii) < eps)
            return false;
        for (int j = i; j < (int)a[i].size(); j++)
            a[i][j] /= aii;
        for (int j = 0; j < n; j++) {
            if (j == i)
                continue;
            for (int k = (int)a[j].size() - 1; k >= i; k--) {
                a[j][k] -= a[j][i] * a[i][k];
            }
        }
    }
    return true;
}

struct linear_regression {
    const int size = feature_count;

    long long instance_count;

    matrix ATA;
    vector ATb;

    linear_regression() {
        reset();
    }

    void reset() {
        instance_count = 0;
        ATA.assign(size, vector(size, 0.0));
        ATb.assign(size, 0.0);
    }

    void add_instance(vector a, double b) {
        assert((int)a.size() == size);
        instance_count ++;
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++)
                ATA[i][j] += a[i] * a[j];
            ATb[i] += a[i] * b;
        }
    }

    void add_other(const linear_regression &o) {
        assert(size == o.size);
        instance_count += o.instance_count;
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++)
                ATA[i][j] += o.ATA[i][j];
            ATb[i] += o.ATb[i];
        }
    }

    vector solve() {
        if (instance_count <= size * 2) {
            PRINT("no enough samples (%lld)\n", instance_count);
        }
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < i; j++) {
                ATA[i][j] = ATA[j][i];
            }
        }
        for (int it = 0; ; ++it) {
            if (it > 10000) {
                DIE("lr failed\n");
            }
            double sum = 1.0;
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    sum += fabs(ATA[i][j]);
            matrix ma = ATA;
            for (int i = 0; i < size; i++) {
                ma[i][i] += sum * it * eps;
                ma[i].push_back(ATb[i]);
            }
            bool res = gaussian_elimination(ma);
            if (!res)
                continue;
            vector ans;
            for (int i = 0; i < size; i++)
                ans.push_back(ma[i][size]);
            return ans;
        }
    }

    void print(FILE *file) {
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++)
                fprintf(file, "%s ", to_string(ATA[i][j]).c_str());
            fprintf(file, "%s\n", to_string(ATb[i]).c_str());
        }
        fprintf(file, "%lld\n", instance_count);
        fprintf(file, "\n");
    }

    void read(FILE *file) {
        reset();
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {
                if (fscanf(file, "%lf", &ATA[i][j]) != 1) {
                    DIE("Invalid solver file (ATA)\n");
                }
            }
            if (fscanf(file, "%lf", &ATb[i]) != 1) {
                DIE("Invalid solver file (ATb)\n");
            }
        }
        if (fscanf(file, "%lld", &instance_count) != 1) {
            DIE("Invalid solver file (instance_count)\n");
        }
    }
};

struct ravu_solver {
    linear_regression lr[quant_angle][quant_strength][quant_coherence][4];

    void reset() {
        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    for (int z = 0; z < 4; z++)
                        lr[i][j][k][z].reset();
    }

    void add_other(const ravu_solver &o) {
        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    for (int z = 0; z < 4; z++)
                        lr[i][j][k][z].add_other(o.lr[i][j][k][z]);
    }

    void train_image(const matrix &hires_image) {
        matrix image = downscale_bicubic(hires_image);
        const int n = 2 * radius - 1;
        const int height = image.size(), width = image[0].size();
        const int height_hi = hires_image.size(), width_hi = hires_image[0].size();
        assert(height * 2 == height_hi && width * 2 == width_hi);
        for (int cx = 0; cx < height; cx++) {
            for (int cy = 0; cy < width; cy++) {
                for (auto &&ma : samplings) {
                    bool found_invalid = false;
                    matrix patch(n, vector(n));
                    vector patch_flatten;
                    for (int i = 0; i < (int)ma.size() && !found_invalid; i++) {
                        for (int j = 0; j < (int)ma[i].size(); j++) {
                            const int tx = cx + ma[i][j].first;
                            const int ty = cy + ma[i][j].second;
                            if (tx < 0 || tx >= height || ty < 0 || ty >= width) {
                                found_invalid = true;
                                break;
                            }
                            patch[i][j] = image[tx][ty];
                            patch_flatten.push_back(image[tx][ty]);
                        }
                    }
                    if (!found_invalid) {
                        hashkey key = get_hashkey(patch);
                        for (int idx = 0; idx < 4; idx ++) {
                            int nidx = get_transposed_index(ma, idx);
                            double hires_value = hires_image[cx * 2 + nidx / 2][cy * 2 + nidx % 2];
                            lr[key.angle][key.strength][key.coherence][idx].add_instance(patch_flatten, hires_value);
                        }
                    }
                }
            }
        }
    }

    void print(FILE *file) {
        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    for (int z = 0; z < 4; z++)
                        lr[i][j][k][z].print(file);
    }

    void read(FILE *file) {
        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    for (int z = 0; z < 4; z++)
                        lr[i][j][k][z].read(file);
    }

};

struct ravu_model {
    std::vector<std::vector<std::vector<matrix>>> lr_weights;

    void reset() {
        lr_weights.assign(quant_angle, std::vector<std::vector<matrix>>(
                quant_strength, std::vector<matrix>(
                    quant_coherence, matrix(4))));
    }

    void solve_from(ravu_solver &solver) {
        reset();
        for (int i = 0; i < quant_angle; i++) {
            for (int j = 0; j < quant_strength; j++) {
                for (int k = 0; k < quant_coherence; k++) {
                    for (int z = 0; z < 4; z++) {
                        lr_weights[i][j][k][z] = solver.lr[i][j][k][z].solve();
                    }
                }
            }
        }
    }

    void print(FILE *file) {
        fprintf(file, "radius = %d\n", radius);
        fprintf(file, "gradient_radius = %d\n", gradient_radius);
        fprintf(file, "quant_angle = %d\n", quant_angle);
        fprintf(file, "quant_strength = %d\n", quant_strength);
        fprintf(file, "quant_coherence = %d\n", quant_coherence);
        fprintf(file, "min_strength = %s\n", to_string(vector(min_strength, min_strength + quant_strength - 1)).c_str());
        fprintf(file, "min_coherence = %s\n", to_string(vector(min_coherence, min_coherence + quant_coherence - 1)).c_str());
        fprintf(file, "gaussian = %s\n", to_string(gaussian).c_str());
        fprintf(file, "model_weights = %s\n", to_string(lr_weights).c_str());
    }

    void read(FILE *file) {
        reset();
        const int buffer_size = quant_angle * quant_strength * quant_coherence * 4 * feature_count * 30;
        static char buffer[buffer_size];
        const char *magic_string = "model_weights = ";
        while (fgets(buffer, buffer_size, file)) {
            if (strncmp(buffer, magic_string, strlen(magic_string)) == 0) {
                std::string line = buffer + strlen(magic_string);
                for (int i = 0; i < (int)line.size(); i++)
                    if (line[i] == '[' || line[i] == ']' || line[i] == ',') {
                        line[i] = ' ';
                    }
                std::istringstream iss(line);
                for (int i = 0; i < quant_angle; i++) {
                    for (int j = 0; j < quant_strength; j++) {
                        for (int k = 0; k < quant_coherence; k++) {
                            for (int z = 0; z < 4; z++) {
                                lr_weights[i][j][k][z].resize(feature_count);
                                for (double &w : lr_weights[i][j][k][z]) {
                                    if (!(iss >> w)) {
                                        DIE("Invalid model file\n");
                                    }
                                }
                            }
                        }
                    }
                }
                return;
            }
        }
        DIE("Invalid model file(no model_weights)\n");
    }

    void read(std::string file_name) {
        FILE *f = fopen(file_name.c_str(), "r");
        if (!f) {
            DIE("failed to open model file '%s'\n", file_name.c_str());
        }
        read(f);
        fclose(f);
    }
};

ravu_solver process(std::string file_name)
{
    const std::string suffix_model = ".ravu-lite" + to_string(radius);
    file_name = remove_suffix(file_name, suffix_model);
    std::string model_file = file_name + suffix_model;
    ravu_solver solver;
    FILE *f = fopen(model_file.c_str(), "r");
    if (f) {
        PRINT("reading %s\n", model_file.c_str());
        solver.read(f);
        fclose(f);
        return solver;
    }
    solver.train_image(read_image(file_name));
    f = fopen(model_file.c_str(), "w");
    if (!f) {
        DIE("failed to write '%s'\n", model_file.c_str());
    }
    PRINT("writing %s\n", model_file.c_str());
    solver.print(f);
    fclose(f);
    return solver;
}

void train(std::vector<std::string> file_names)
{
    ravu_solver solver;
    for (std::string file_name : file_names) {
        ravu_solver single_solver = process(file_name);
        solver.add_other(single_solver);
    }
    ravu_model my_model;
    PRINT("final solving\n");
    my_model.solve_from(solver);
    my_model.print(stdout);
}

double psnr(const matrix &a, const matrix &b)
{
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    double sumsq = 0;
    int num = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        for (int j = 0; j < (int)a[0].size(); j++) {
            double diff = a[i][j] - b[i][j];
            sumsq += diff * diff;
            num ++;
        }
    }
    return 10 * log10(1.0 / (sumsq / num));
}

matrix upscale_ravu(const ravu_model &model, const matrix &lowres_image)
{
    const int height = lowres_image.size(), width = lowres_image[0].size();
    const int height_hi = height * 2, width_hi = width * 2;

    matrix image(height_hi, vector(width_hi, 0.0));

    const std::vector<std::vector<std::pair<int,int>>> &ma = samplings[0];
    for (int i = 0; i < height; i ++) {
        for (int j = 0; j < width; j ++) {
            vector patch_flatten;
            matrix patch;
            for (auto &&row : ma) {
                patch.emplace_back();
                for (auto &&cell : row) {
                    int x = std::min(height - 1, std::max(0, i + cell.first));
                    int y = std::min(width - 1, std::max(0, j + cell.second));
                    patch.back().push_back(lowres_image[x][y]);
                    patch_flatten.push_back(lowres_image[x][y]);
                }
            }
            const hashkey key = get_hashkey(patch);
            for (int idx = 0; idx < 4; idx ++) {
                const vector &w = model.lr_weights[key.angle][key.strength][key.coherence][idx];
                double sum = 0;
                for (int k = 0; k < feature_count; k++) {
                    sum += w[k] * patch_flatten[k];
                }
                image[i * 2 + idx / 2][j * 2 + idx % 2] = sum;
            }
        }
    }

    return image;
}

double predict(std::string weights, std::vector<std::string> file_names)
{
    ravu_model my_model;
    my_model.read(weights);
    double sum_psnr = 0;
    for (std::string file_name : file_names) {
        matrix hires_image = read_image(file_name);
        matrix lowres_image = downscale_bicubic(hires_image);
        matrix bicubic_upscaled = upscale_bicubic(lowres_image);
        matrix ravu_upscaled = upscale_ravu(my_model, lowres_image);

        double bicubic_psnr = psnr(hires_image, bicubic_upscaled);
        double ravu_psnr = psnr(hires_image, ravu_upscaled);
        PRINT("bicubic: %.6lf, ravu: %.6lf\n", bicubic_psnr, ravu_psnr);
        sum_psnr += ravu_psnr - bicubic_psnr;
    }
    double avg_psnr = sum_psnr / file_names.size();
    PRINT("avg psnr improve: %.6lf\n", avg_psnr);

    return avg_psnr;
}

int main(int argc, char *argv[])
{
    prepare_gaussian_2d();
    prepare_samplings();
    int argv_ptr = 1;
    if (argv_ptr + 1 >= argc) {
        PRINT("Usage: %s [train|process|predict weights.py] file1.pnm [file2.pnm..]\n", argv[0]);
        return -1;
    }
    std::string command = argv[argv_ptr];
    std::vector<std::string> file_names(argv + argv_ptr + 1, argv + argc);
    if (command == "train") {
        train(file_names);
    } else if (command == "process") {
        for (std::string file_name : file_names)
            process(file_name);
    } else if (command == "predict") {
        std::string weights = file_names[0];
        file_names.erase(file_names.begin());
        predict(weights, file_names);
    } else {
        DIE("Unknown command: '%s'\n", command.c_str());
    }
}

