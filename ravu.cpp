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
const int quant_strength = 9;
const int quant_coherence = 3;

const double min_strength[quant_strength - 1] = {0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128};
const double min_coherence[quant_coherence - 1] = {0.25, 0.5};

const int feature_count = (2 * radius) * (2 * radius);

struct hashkey {
    int angle;
    int strength;
    int coherence;

    hashkey(int angle, int strength, int coherence) : angle(angle), strength(strength), coherence(coherence) { }
};

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

template<typename T>
void vector_trunc_range(std::vector<T> &x, int l, int r)
{
    assert(l < r);
    if (r < (int)x.size())
        x.resize(r);
    if (l > 0)
        x.erase(x.begin(), x.begin() + l);
}

std::string remove_suffix(const std::string &file_name, const std::string &suffix)
{
    if (file_name.size() > suffix.size() &&
            file_name.substr(file_name.size() - suffix.size()) == suffix) {
        return file_name.substr(0, file_name.size() - suffix.size());
    }
    return file_name;
}

matrix read_image(const std::string &file_name, bool trunc=true)
{
    // for i in *.jpg; do convert -depth 16 -filter Mitchell -resize 75% "$i" -colorspace gray -compress none "$(basename "$i" .jpg).pnm"; done
    // for i in *.png; do convert -depth 16 "$i" -colorspace gray -compress none "$(basename "$i" .png).pnm"; done
    PRINT("reading %s\n", file_name.c_str());
    FILE* image_file = fopen(file_name.c_str(), "r");
    if (!image_file) {
        DIE("file '%s' not found\n", file_name.c_str());
    }
    char magic_string[10];
    fgets(magic_string, sizeof(magic_string), image_file);
    if (strncmp(magic_string, "P2", 2) != 0) {
        DIE("file '%s' is not a valid PGM file (ASCII)\n", file_name.c_str());
    }

    int width, height, max_value;
    if (fscanf(image_file, "%d%d%d", &width, &height, &max_value) != 3) {
        DIE("file '%s' is not a valid PGM file (ASCII)\n", file_name.c_str());
    }

    matrix ret(height, vector(width));

    int minx = height, maxx = 0;
    int miny = width, maxy = 0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int value;
            if (fscanf(image_file, "%d", &value) != 1 || value < 0 || value > max_value) {
                DIE("file '%s' is not a valid PGM file (ASCII)\n", file_name.c_str());
            }
            if (value > 0) {
                minx = std::min(minx, i);
                maxx = std::max(maxx, i);
                miny = std::min(miny, j);
                maxy = std::max(maxy, j);
            }
            ret[i][j] = (double)value / max_value;
        }
    }
    fclose(image_file);
    if (trunc) {
        if (minx <= maxx && miny <= maxy) {
            vector_trunc_range(ret, minx, maxx + 1);
            for (auto &&row : ret)
                vector_trunc_range(row, miny, maxy + 1);
        }
    }
    return ret;
}

matrix gaussian;

void prepare_gaussian_2d()
{
    const double sigma = 2;
    const int n = 2 * gradient_radius;
    const double center = gradient_radius - 0.5;
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
    const int border = 2 * radius - 1;
    std::vector<std::vector<std::pair<int,int>>> ma;
    for (int x = -border; x <= border; x += 2) {
        ma.emplace_back();
        for (int y = -border; y <= border; y += 2) {
            ma.back().emplace_back(x, y);
        }
    }
    for (int scale = 0; scale < 2; scale++) {
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
        for (auto &&row : ma) {
            for (auto &&cell : row) {
                cell = std::make_pair((cell.first + cell.second) / 2, (cell.first - cell.second) / 2);
            }
        }
    }
}

hashkey get_hashkey(const matrix &patch)
{
    const int n = 2 * radius;
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
            else if (i == 1 || i == n - 2)
                gx = (patch[i + 1][j] - patch[i - 1][j]) / 2.0;
            else
                gx = (-patch[i + 2][j] + 8 * patch[i + 1][j] - 8 * patch[i - 1][j] + patch[i - 2][j]) / 12.0;
            if (j == 0)
                gy = patch[i][j + 1] - patch[i][j];
            else if (j == n - 1)
                gy = patch[i][j] - patch[i][j - 1];
            else if (j == 1 || j == n - 2)
                gy = (patch[i][j + 1] - patch[i][j - 1]) / 2.0;
            else
                gy = (-patch[i][j + 2] + 8 * patch[i][j + 1] - 8 * patch[i][j - 1] + patch[i][j - 2]) / 12.0;
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
    linear_regression lr[quant_angle][quant_strength][quant_coherence];

    void reset() {
        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    lr[i][j][k].reset();
    }

    void add_other(const ravu_solver &o) {
        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    lr[i][j][k].add_other(o.lr[i][j][k]);
    }

    void train_image(const matrix &image) {
        const int height = image.size(), width = height == 0 ? 0 : image[0].size();
        for (int cx = 0; cx < height; cx++) {
            for (int cy = 0; cy < width; cy++) {
                for (auto &&ma : samplings) {
                    bool found_invalid = false;
                    matrix patch(2 * radius, vector(2 * radius));
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
                        lr[key.angle][key.strength][key.coherence].add_instance(patch_flatten, image[cx][cy]);
                    }
                }
            }
        }
    }

    void print(FILE *file) {
        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    lr[i][j][k].print(file);
    }

    void read(FILE *file) {
        for (int i = 0; i < quant_angle; i++)
            for (int j = 0; j < quant_strength; j++)
                for (int k = 0; k < quant_coherence; k++)
                    lr[i][j][k].read(file);
    }

};

struct ravu_model {
    std::vector<std::vector<std::vector<vector>>> lr_weights;

    void reset() {
        lr_weights.assign(quant_angle, std::vector<std::vector<vector>>(
                quant_strength, std::vector<vector>(
                    quant_coherence)));
    }

    void solve_from(ravu_solver &solver) {
        reset();
        for (int i = 0; i < quant_angle; i++) {
            for (int j = 0; j < quant_strength; j++) {
                for (int k = 0; k < quant_coherence; k++) {
                    lr_weights[i][j][k] = solver.lr[i][j][k].solve();
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
        const int buffer_size = quant_angle * quant_strength * quant_coherence * feature_count * 30;
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
                            lr_weights[i][j][k].resize(feature_count);
                            for (double &w : lr_weights[i][j][k]) {
                                if (!(iss >> w)) {
                                    DIE("Invalid model file\n");
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
    const std::string suffix = ".ravu" + to_string(radius);
    file_name = remove_suffix(file_name, suffix);
    std::string out_file = file_name + suffix;
    ravu_solver solver;
    FILE *f = fopen(out_file.c_str(), "r");
    if (f) {
        PRINT("reading %s\n", out_file.c_str());
        solver.read(f);
        fclose(f);
        return solver;
    }
    solver.train_image(read_image(file_name));
    f = fopen(out_file.c_str(), "w");
    if (!f) {
        DIE("failed to write '%s'\n", out_file.c_str());
    }
    PRINT("writing %s\n", out_file.c_str());
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

void predict(std::string weights, std::vector<std::string> file_names)
{
    ravu_model my_model;
    my_model.read(weights);
    double sum_psnr = 0;
    int count_psnr = 0;
    for (std::string file_name : file_names) {
        matrix image = read_image(file_name);
        int height = image.size(), width = height == 0 ? 0 : image[0].size();
        matrix lowres = image;
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                lowres[i][j] = image[i & ~1][j & ~1];
        for (int pass = 0; pass < 2; pass++) {
            std::vector<std::vector<std::pair<int,int>>> ma;
            if (pass == 0) {
                ma = samplings[0];
            } else {
                ma = samplings[samplings.size() / 2];
            }
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    if (pass == 0) {
                        if (i % 2 == 0 || j % 2 == 0)
                            continue;
                    } else {
                        if ((i + j) % 2 == 0)
                            continue;
                    }
                    vector patch_flatten;
                    matrix patch;
                    for (auto &&row : ma) {
                        patch.emplace_back();
                        for (auto &&cell : row) {
                            int x = i + cell.first;
                            int y = j + cell.second;
                            if (x < 0 || x >= height || y < 0 || y >= width) {
                                x = std::min(height - 1, std::max(0, x)) & ~1;
                                y = std::min(width - 1, std::max(0, y)) & ~1;
                            }
                            patch.back().push_back(lowres[x][y]);
                            patch_flatten.push_back(lowres[x][y]);
                        }
                    }
                    const hashkey key = get_hashkey(patch);
                    const vector &w = my_model.lr_weights[key.angle][key.strength][key.coherence];
                    double sum = 0;
                    for (int k = 0; k < feature_count; k++) {
                        sum += w[k] * patch_flatten[k];
                    }
                    lowres[i][j] = sum;
                }
            }
        }
        double sumsq = 0, sum = 0;
        long long count = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double diff = image[i][j] - lowres[i][j];
                count ++;
                sumsq += diff * diff;
                sum += std::abs(diff);
            }
        }
        if (count == 0)
            continue;
        double mean_squared_error = sumsq / count;
        double psnr = 10 * log10(1.0  / mean_squared_error);
        PRINT("psnr: %.5lf\n", psnr);
        sum_psnr += psnr;
        count_psnr ++;
    }
    if (count_psnr >= 2) {
        PRINT("mean psnr: %.5lf\n", sum_psnr / count_psnr);
    }
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

