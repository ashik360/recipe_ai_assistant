import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'recipe_data.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      showSemanticsDebugger: false,
      title: 'Visual Recipe Assistant',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        fontFamily: 'Poppins',
        scaffoldBackgroundColor: const Color(0xFFF9F9F9),
        cardTheme: CardThemeData(
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _image;
  bool _isLoading = false;
  String? _detectedIngredient;
  double? _confidence;
  List<String> _recipes = [];
  Interpreter? _interpreter;
  List<String> _labels = [];
  final ImagePicker _picker = ImagePicker();
  String _debugInfo = 'Initializing...';

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    setState(() {
      _isLoading = true;
      _debugInfo = 'Loading Teachable Machine model...';
    });

    try {
      // Load the quantized model from Teachable Machine
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      
      // Load labels (Teachable Machine format)
      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();

      // Get tensor information
      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);

      setState(() {
        _debugInfo =
            'Teachable Machine model loaded!\n'
            'Labels: ${_labels.length} classes\n'
            'Input: ${inputTensor.shape} (${inputTensor.type})\n'
            'Output: ${outputTensor.shape} (${outputTensor.type})\n'
            'Model type: Quantized (uint8)';
      });
    } catch (e) {
      debugPrint('Error loading model: $e');
      setState(() {
        _debugInfo = 'Error loading model: $e\nMake sure model.tflite and labels.txt are in assets/';
      });
    }

    setState(() => _isLoading = false);
  }

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _detectedIngredient = null;
        _confidence = null;
        _recipes = [];
        _debugInfo = 'Image selected, processing with Teachable Machine model...';
      });
      _classifyImage();
    }
  }

  void _resetSelection() {
    setState(() {
      _image = null;
      _detectedIngredient = null;
      _confidence = null;
      _recipes = [];
      _debugInfo = 'Ready to classify ingredients';
    });
  }

  // Clean label from Teachable Machine format (e.g., "0 tomato" -> "tomato")
  String _cleanLabel(String rawLabel) {
    // Remove leading numbers and spaces (e.g., "0 tomato" -> "tomato")
    String cleaned = rawLabel.replaceAll(RegExp(r'^\d+\s*'), '');
    
    // Remove any trailing whitespace
    cleaned = cleaned.trim();
    
    // Convert to lowercase for consistency
    cleaned = cleaned.toLowerCase();
    
    return cleaned;
  }

  Future<void> _classifyImage() async {
    if (_image == null || _interpreter == null) return;

    setState(() {
      _isLoading = true;
      _debugInfo = 'Running Teachable Machine inference...';
    });

    try {
      // Preprocess image for Teachable Machine (224x224, uint8)
      final imageBytes = await _image!.readAsBytes();
      final input = _preprocessImageForTeachableMachine(imageBytes);

      // Prepare output buffer for quantized model
      final outputTensor = _interpreter!.getOutputTensor(0);
      final outputShape = outputTensor.shape;
      final outputLength = outputShape.reduce((a, b) => a * b);

      // Teachable Machine quantized models typically output uint8
      final output = Uint8List(outputLength);

      // Run inference
      final stopwatch = Stopwatch()..start();
      _interpreter!.run(input, output);
      stopwatch.stop();

      // Convert quantized output to probabilities
      final probabilities = _dequantizeOutput(output);

      // Find the class with highest probability
      final maxValue = probabilities.reduce(max);
      final maxIndex = probabilities.indexOf(maxValue);

      // Validate results
      if (maxIndex >= _labels.length) {
        setState(() {
          _debugInfo = 'Error: Invalid class index $maxIndex (max: ${_labels.length - 1})';
        });
        return;
      }

      if (maxValue < 0.1) {
        setState(() {
          _debugInfo = 'Low confidence: ${(maxValue * 100).toStringAsFixed(1)}%. Try a clearer image.';
        });
        return;
      }

      // Get detected ingredient and recipes
      final rawLabel = _labels[maxIndex];
      final cleanedLabel = _cleanLabel(rawLabel);
      final recipes = RecipeData.getRecipes(cleanedLabel);

      setState(() {
        _detectedIngredient = cleanedLabel;
        _confidence = maxValue;
        _recipes = recipes;
        _debugInfo =
            'Teachable Machine classification complete!\n'
            'Time: ${stopwatch.elapsedMilliseconds}ms\n'
            'Raw label: $rawLabel\n'
            'Cleaned: $cleanedLabel\n'
            'Confidence: ${(maxValue * 100).toStringAsFixed(1)}%\n'
            'Recipes found: ${recipes.length}';
      });
    } catch (e) {
      debugPrint('Classification error: $e');
      setState(() {
        _debugInfo = 'Classification error: $e';
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Preprocess image for Teachable Machine quantized model
  Uint8List _preprocessImageForTeachableMachine(Uint8List imageBytes) {
    final img.Image? image = img.decodeImage(imageBytes);
    if (image == null) {
      throw Exception("Unable to decode image");
    }

    // Teachable Machine expects 224x224 RGB images
    const int inputSize = 224;

    // Crop to square (center crop)
    final int cropSize = min(image.width, image.height);
    final int offsetX = (image.width - cropSize) ~/ 2;
    final int offsetY = (image.height - cropSize) ~/ 2;
    
    final img.Image cropped = img.copyCrop(
      image,
      x: offsetX,
      y: offsetY,
      width: cropSize,
      height: cropSize,
    );

    // Resize to 224x224
    final img.Image resized = img.copyResize(
      cropped,
      width: inputSize,
      height: inputSize,
    );

    // Convert to uint8 buffer (1, 224, 224, 3)
    final buffer = Uint8List(1 * inputSize * inputSize * 3);
    int bufferIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        // Store as RGB uint8 values (0-255)
        buffer[bufferIndex++] = pixel.r.toInt();
        buffer[bufferIndex++] = pixel.g.toInt();
        buffer[bufferIndex++] = pixel.b.toInt();
      }
    }

    return buffer;
  }

  // Convert quantized uint8 output to probabilities
  List<double> _dequantizeOutput(Uint8List quantizedOutput) {
    // Teachable Machine quantized models typically use:
    // - Scale: 1/255 (or similar)
    // - Zero point: 0
    // For simplicity, we'll normalize uint8 values to 0-1 range
    final probabilities = quantizedOutput.map((value) => value / 255.0).toList();
    
    // Apply softmax to convert to proper probabilities
    final maxVal = probabilities.reduce(max);
    final expValues = probabilities.map((x) => exp(x - maxVal)).toList();
    final sumExp = expValues.reduce((a, b) => a + b);
    
    return expValues.map((x) => x / sumExp).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            Expanded(
              child: _isLoading
                  ? _buildLoadingIndicator()
                  : _buildMainContent(),
            ),
            _buildDebugInfo(),
            _buildActionButtons(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.teal.shade700,
        borderRadius: const BorderRadius.only(
          bottomLeft: Radius.circular(24),
          bottomRight: Radius.circular(24),
        ),
      ),
      child: Row(
        children: [
          const Icon(Icons.restaurant_menu, color: Colors.white, size: 32),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              'Visual Recipe Assistant',
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: _resetSelection,
            tooltip: 'Reset',
          ),
        ],
      ),
    );
  }

  Widget _buildLoadingIndicator() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const CircularProgressIndicator(color: Colors.teal, strokeWidth: 4),
          const SizedBox(height: 20),
          Text('Processing with Teachable Machine...', 
               style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 20),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Text(
              _debugInfo,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 12, color: Colors.grey),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMainContent() {
    return SingleChildScrollView(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            _buildImagePreview(),
            if (_detectedIngredient != null) _buildResultsCard(),
            if (_recipes.isNotEmpty) _buildRecipesList(),
            if (_image == null) _buildEmptyState(),
          ],
        ),
      ),
    );
  }

  Widget _buildImagePreview() {
    return Container(
      height: 280,
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Stack(
        children: [
          if (_image != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Image.file(
                _image!,
                fit: BoxFit.cover,
                width: double.infinity,
                height: double.infinity,
              ),
            )
          else
            Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.photo_camera_back,
                    size: 80,
                    color: Colors.grey.shade300,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Upload ingredient image',
                    style: Theme.of(context)
                        .textTheme
                        .bodyLarge
                        ?.copyWith(color: Colors.grey),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Model Trained By Shuvo ID: 059\nPowered by Teachable Machine',
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: Colors.grey.shade400),
                  ),
                ],
              ),
            ),
          if (_image != null)
            Positioned(
              top: 10,
              right: 10,
              child: IconButton(
                icon: const Icon(Icons.close, color: Colors.white),
                onPressed: _resetSelection,
                style: IconButton.styleFrom(backgroundColor: Colors.black54),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildResultsCard() {
    return Padding(
      padding: const EdgeInsets.only(top: 24),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.teal.withOpacity(0.1),
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.check_circle,
                  color: Colors.teal,
                  size: 36,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Detected Ingredient',
                      style: Theme.of(context)
                          .textTheme
                          .bodySmall
                          ?.copyWith(color: Colors.grey),
                    ),
                    Text(
                      _detectedIngredient!,
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    const SizedBox(height: 4),
                    LinearProgressIndicator(
                      value: _confidence,
                      backgroundColor: Colors.grey.shade200,
                      color: Colors.teal,
                      minHeight: 8,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Confidence: ${(_confidence! * 100).toStringAsFixed(1)}%',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRecipesList() {
    return Padding(
      padding: const EdgeInsets.only(top: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'Recommended Recipes',
                style: Theme.of(context)
                    .textTheme
                    .titleLarge
                    ?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(width: 10),
              Text(
                '(${_recipes.length} found)',
                style: TextStyle(color: Colors.teal.shade700),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ..._recipes.map(
            (recipe) => Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Container(
                        width: 4,
                        height: 40,
                        decoration: BoxDecoration(
                          color: Colors.teal,
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Text(
                          recipe,
                          style: Theme.of(context).textTheme.bodyLarge,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Padding(
      padding: const EdgeInsets.only(top: 40),
      child: Column(
        children: [
          Icon(Icons.fastfood, size: 80, color: Colors.teal.withOpacity(0.3)),
          const SizedBox(height: 24),
          Text(
            'Discover Recipes from Ingredients',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 16),
          Text(
            'Upload a photo of your ingredients to find delicious recipes',
            textAlign: TextAlign.center,
            style: Theme.of(context)
                .textTheme
                .bodyLarge
                ?.copyWith(color: Colors.grey),
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.teal.withOpacity(0.1),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              'Model Trained By Shuvo ID: 059\nPowered by Teachable Machine',
              style: Theme.of(context)
                  .textTheme
                  .bodyMedium
                  ?.copyWith(color: Colors.teal.shade700),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDebugInfo() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: Colors.grey.shade100,
      child: Row(
        children: [
          const Icon(Icons.info_outline, size: 16, color: Colors.grey),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              _debugInfo,
              style: const TextStyle(fontSize: 12, color: Colors.grey),
              overflow: TextOverflow.ellipsis,
              maxLines: 4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons() {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          FloatingActionButton.extended(
            heroTag: 'gallery',
            onPressed: () => _pickImage(ImageSource.gallery),
            icon: const Icon(Icons.photo_library, color: Colors.white,),
            label: const Text('Gallery', style: TextStyle(color: Colors.white),),
            backgroundColor: Colors.teal,
          ),
          FloatingActionButton.extended(
            heroTag: 'camera',
            onPressed: () => _pickImage(ImageSource.camera),
            icon: const Icon(Icons.camera_alt, color: Colors.white,),
            label: const Text('Camera', style: TextStyle(color: Colors.white),),
            backgroundColor: Colors.teal.shade700,
          ),
        ],
      ),
    );
  }
}