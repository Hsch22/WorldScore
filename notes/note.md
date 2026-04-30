1. 总览
WorldScore: A Unified Evaluation Benchmark for World Generation 提供一套统一流程，用来评估不同“世界生成”模型生成连续视觉世界的能力。

这里的“世界生成”可以包括三类模型：
- 3D scene generation：例如 WonderJourney、WonderWorld、SceneScape、Text2Room、LucidDreamer、InvisibleStitch。
- 4D generation：例如 4D-fy。
- Video generation：例如 CogVideoX、VideoCrafter、DynamiCrafter、T2V-Turbo、Vchitect、EasyAnimate、Allegro、Gen-3、Minimax、LTX-Video、Wan 等。

Benchmark 的核心问题是：给模型一个初始图片、文本描述和相机/运动要求后，模型生成的视频是否真的表现出了一个稳定、可探索、符合指令的世界。

它不只评估单帧画质，也评估：
- 相机是否按要求运动；
- 场景内容是否符合文本和目标物体；
- 生成世界是否有 3D 一致性；
- 连续帧是否稳定、连贯；
- 风格是否一致；
- 动态物体是否真的在动；
- 运动是否足够明显、准确和平滑。

代码中最终会汇总两个主分数：
- WorldScore-Static：静态世界生成能力分数。
- WorldScore-Dynamic：静态能力加动态能力后的综合分数。

2. 整体评测流程
2.1 数据输入
数据集由 download.py 从 Hugging Face 下载到：
$DATA_PATH/WorldScore-Dataset

数据分为两类：
- static：静态世界生成任务。
- dynamic：动态世界生成任务。

加载入口在worldscore/benchmark/helpers/__init__.py，其中 GetHelpers(model_name, visual_movement, json_file="") 会根据 visual_movement 自动读取：
$DATA_PATH/WorldScore-Dataset/static/static.json
$DATA_PATH/WorldScore-Dataset/dynamic/dynamic.json

每个数据点中包含初始图片、文本 prompt、视觉风格、相机路径、场景类别、动态物体 mask等信息。

2.2 模型生成
生成入口是world_generators/generate_videos.py

它会：
1. 根据 config/model_configs/<model_name>.yaml 读取模型的输出目录、分辨率、帧数、fps、生成类型等。
2. 根据 world_generators/configs/<model_name>.yaml 用 Hydra 实例化具体模型 wrapper。
3. 调用 GetHelpers 得到 benchmark 数据和 helper。
4. 对每个数据点调用对应 adapter，把 benchmark 数据转成模型输入。
5. 调用模型的 generate_video(prompt, image_path)。
6. 保存生成帧、视频和元数据。

生成结果会保存到：
$MODEL_PATH/<model_repo>/worldscore_output/static/...
$MODEL_PATH/<model_repo>/worldscore_output/dynamic/...

每个样本目录通常包含：
frames/
videos/output.mp4
input_image.png
image_data.json
camera_data.json    # static 任务需要
time.txt

2.3 模型适配
模型适配层在：
worldscore/benchmark/helpers/adapters/

代码把模型分成三类：
type2model = {
    "threedgen": [...],
    "fourdgen": [...],
    "videogen": [...],
}

具体在：
worldscore/benchmark/utils/modeltype.py

对 video generation 模型，adapter 根据 generate_type 选择：
- adapter_i2v.py: image-to-video. 
- adapter_t2v.py: text-to-video. 

对 3D generation 模型，则有模型专属 adapter，例如：
- adapter_wonderjourney.py
- adapter_wonderworld.py
- adapter_scenescape.py

这些 adapter 的作用是统一输出模型需要的输入格式。例如 I2V 模型会拿到image_path，prompt_list
而 WonderJourney 这类 3D 模型会拿到start_keyframe，inpainting_prompt_list，cameras，cameras_interp

2.4 相机路径生成
静态任务中，benchmark 会生成标准相机轨迹作为 ground truth。实现位置：
worldscore/benchmark/helpers/camera_generator.py

支持的相机动作定义在：
worldscore/benchmark/utils/utils.py

包括：
- push_in
- pull_out
- move_left
- move_right
- orbit_left
- orbit_right
- pan_left
- pan_right
- fixed
每个动作都有：
- scenenum：动作对应的新场景数量。
- prompt：写入文本 prompt 的相机动作描述。
- layout_type：动作类型，例如 intra、inter、camera fix。
CameraGen 会先生成关键帧相机，再通过旋转球面插值和位移线性插值得到每一帧的相机矩阵，最后写入：
camera_data.json
评测相机控制和 3D 几何一致性时会用到这些相机数据。

3. Static 任务评测：测静态世界生成能力
Static 任务评估的是：在场景本身保持静止的情况下，模型是否能根据相机运动扩展/探索一个稳定的世界。
代码中的 static 指标列表在 worldscore/run_evaluate.py：
worldscore_list = {
    "static": [
        "camera_control",
        "object_control",
        "content_alignment",
        "3d_consistency",
        "photometric_consistency",
        "style_consistency",
        "subjective_quality",
    ]
}

3.1 Camera Control：相机控制能力
测什么：
模型生成的视频是否遵循指定相机路径，例如 pan left、move left、pull out、orbit right 等。
为什么重要：
世界生成模型不能只生成看起来连贯的视频，还应该能让观察者按照指定轨迹移动视角。如果 prompt 要求“向左平移”，视频却只是物体变形或画面随机抖动，那么世界可控性是不合格的。

流程如下：
1. 输入生成视频的所有帧。
2. 用 DROID-SLAM 从生成帧中估计相机轨迹。
3. 读取 camera_data.json 中的 ground-truth 插值相机轨迹。
4. 把预测相机和 GT 相机转换到统一坐标系。
5. 对预测平移做尺度对齐。
6. 计算：
  - 平均旋转误差；
  - 平均平移误差。


3.2 Object Control：目标物体控制能力
测什么：
生成结果中是否出现了 prompt 中要求的关键物体。
为什么重要：
世界生成不只是生成背景，还要能保留或生成指定实体。例如 prompt 中要求某个房间有床、灯、桌子，模型应该在相应场景中生成这些对象。

流程如下：
1. 对每个场景取 anchor frame。
2. 根据 content_list 或文本 prompt 提取目标物体。
3. 使用 GroundingDINO 检测图像中的目标。
4. 对检测结果做字符串标准化和名词匹配。
5. 计算目标物体被检测到的比例。
分数是 detection success rate，范围大致为[0，1]

3.3 Content Alignment：文本内容对齐能力
测什么：
生成图像是否和文本描述语义一致。
为什么重要：
即使相机轨迹正确，如果生成出来的场景内容和 prompt 不匹配，也不能算符合世界生成任务。

流程如下：
1. 对每个场景取 anchor frame。
2. 读取该场景对应的文本 prompt。
3. 用 CLIPScore 计算图像和文本之间的相似度。
4. 对多个场景求平均。
CLIPScore 原始值越高越好。代码中使用经验均值、标准差和 z-score 范围进行归一化。

注意：在 Evaluator.process_batch 中，如果相机动作属于 intra 类型，content_alignment 会被跳过，因为 intra movement 更偏向同一场景内部观察，不一定要求新内容出现。

3.4 3D Consistency：三维一致性
测什么：
生成视频是否表现出稳定的 3D 结构。
为什么重要：
世界生成应当维持空间结构一致。比如相机移动时，墙、桌子、门窗应该符合几何关系，而不是每一帧都重新生成出不一致的形状。

流程如下：
1. 输入整段生成视频帧。
2. 使用 DROID-SLAM 进行 SLAM 跟踪。
3. DROID-SLAM 在终止时返回 valid reprojection errors。
4. 对有效重投影误差取平均。
重投影误差越小，说明跨帧几何越一致。

3.5 Photometric Consistency：光度一致性
测什么：
连续帧之间的像素/运动关系是否自洽。
为什么重要：
如果生成视频有明显闪烁、局部跳变或前后帧无法通过合理光流对应起来，说明世界不稳定。

流程如下：
1. 对每一对相邻帧 (frame_t, frame_t+1)：
  - 用 SEA-RAFT 估计正向光流；
  - 再估计反向光流。
2. 通过正反向光流做 cycle consistency 检查。
3. 计算 average endpoint error，也就是光流回环后的平均端点误差。
4. 对所有相邻帧取平均。

3.6 Style Consistency：风格一致性
测什么：
生成的新场景是否和初始图像保持一致的视觉风格。
为什么重要：
世界扩展过程中，画面风格不应突然从照片变成插画，或从写实变成卡通。

流程如下：
1. 读取 input_image.png 作为 reference image。
2. 读取生成视频中后续场景的 anchor frames。
3. 用 VGG19 提取多层卷积特征。
4. 对特征计算 Gram Matrix。
5. 计算 reference image 和 generated image 的 Gram Matrix MSE。
Gram Matrix 距离越小，说明风格越一致。

3.7 Subjective Quality：主观视觉质量
测什么：
生成图像本身是否有较好的感知质量和审美质量。
为什么重要：
即使几何和内容正确，如果图像严重模糊、破碎、噪声大，也不是高质量世界生成。

评测流程：
1. 对每个场景取 anchor frame。
2. 用 CLIP-IQA+ 估计图像质量。
3. 用 LAION aesthetic 模型估计审美分。
4. 两个指标分别归一化，再在 subjective_quality 下汇总。
这两个指标都是越高越好。

4. Dynamic 任务评测：测动态世界生成能力
Dynamic 任务评估的是：在有目标物体运动要求时，模型是否能生成合理的动态内容。
代码中的 dynamic 指标列表在 worldscore/run_evaluate.py：
worldscore_list = {
    "dynamic": [
        "motion_accuracy",
        "motion_magnitude",
        "motion_smoothness",
    ]
}

4.1 Motion Accuracy：运动准确性
测什么：
目标物体是否真的发生了符合任务要求的运动，而不是只有背景或相机在动。
为什么重要：
动态世界生成不只是让画面整体有光流，还要让指定对象动起来。如果 prompt 要求物体运动，但实际只有背景在变，模型应被扣分。

流程如下：
1. 读取动态任务提供的初始 mask。
2. 使用 SAM2 video predictor 在整段生成视频中传播目标 mask，得到每一帧的目标区域。
3. 对相邻帧使用 SEA-RAFT 计算 optical flow。
4. 分别统计：
  - 目标区域的 flow magnitude；
  - 背景区域的 flow magnitude。
5. 计算目标区域运动强度和背景运动强度的差值。
代码核心分数是：
max(object_flow_magnitude) - max(background_flow_magnitude)
再对所有相邻帧求平均。
分数越高，说明目标物体相对背景的运动越明显，动态控制越好。

4.2 Motion Magnitude：运动幅度
测什么：
视频整体是否具有足够的运动量。
为什么重要：
有些模型会生成几乎静止的视频。画面质量可能不错，但动态任务中这不符合要求。

流程如下：
1. 对所有相邻帧使用 SEA-RAFT 估计 optical flow。
2. 计算每个像素的 flow magnitude。
3. 对每对帧取 flow magnitude 的中位数。
4. 对整段视频求平均。
原始分数越高，说明运动幅度越大。代码使用经验统计值进行归一化。

4.3 Motion Smoothness：运动平滑性
测什么：
生成视频的运动是否平滑，是否存在跳帧、闪烁、突然形变。
为什么重要：
动态世界生成不仅需要物体动，还需要动作连续自然。

流程如下：
1. 把生成视频分成偶数帧和奇数帧。
2. 使用 VFIMamba 根据前后偶数帧插值预测中间帧。
3. 把模型预测的中间帧和真实生成视频中的奇数帧比较。
4. 计算三个指标：
  - MSE; 
  - SSIM; 
  - LPIPS. 
返回结果是：
(mse_score, ssim_score, lpips_score)

5. 分数归一化与汇总
5.1 单指标归一化
指标配置集中定义在：
worldscore/benchmark/utils/utils.py
变量名aspect_info，每个 aspect 会指定：
- metric 名称；
- metric 类型；
- 经验最大/最小值，或均值/标准差；
- higher_is_better. 
归一化逻辑在worldscore/benchmark/helpers/evaluator.py
主要函数normalize_score(...)和renormalize_score(...)

大致有两种归一化方式：
1. 对有经验上下界的指标，用 min/max 映射到 [0, 1]。
2. 对 CLIPScore、CLIP-IQA、aesthetic、motion accuracy、optical flow 这类指标，用均值、标准差和 z-score 范围做归一化。

5.2 每个样本的结果
每个样本评测后会写入：
evaluation.json

结构大致是：
{
  "camera_control": {
    "camera_error": {
      "score": [...],
      "score_normalized": ...
    }
  },
  "content_alignment": {
    "clip_score": {
      "score": ...,
      "score_normalized": ...
    }
  }
}

5.3 模型总分
最终汇总在：
worldscore/run_evaluate.py

函数：
calculate_worldscore(...)
它会遍历模型输出目录中的所有 evaluation.json，对每个 aspect 下的 metric 做平均。
Static 总分：
WorldScore-Static = mean([
    camera_control,
    object_control,
    content_alignment,
    3d_consistency,
    photometric_consistency,
    style_consistency,
    subjective_quality,
])

Dynamic 总分：
WorldScore-Dynamic = mean([
    camera_control,
    object_control,
    content_alignment,
    3d_consistency,
    photometric_consistency,
    style_consistency,
    subjective_quality,
    motion_accuracy,
    motion_magnitude,
    motion_smoothness,
])
注意：代码中 threedgen 模型只跑 static；video generation 和 4D generation 模型会跑 static 和 dynamic。
