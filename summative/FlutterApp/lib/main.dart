import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// Public address of the deployed FastAPI service.
const String apiBaseUrl = 'https://gbv-prevalence-api.onrender.com';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GBV Prevalence Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.deepPurple, useMaterial3: true),
      home: const PredictionPage(),
    );
  }
}

enum FieldType { text, integer, decimal }

/// One input of the model: the JSON key the API expects, a label and a hint.
class FieldSpec {
  const FieldSpec(
    this.key,
    this.label,
    this.hint, [
    this.type = FieldType.text,
  ]);

  final String key;
  final String label;
  final String hint;
  final FieldType type;
}

const List<FieldSpec> kFields = [
  FieldSpec('CountryName', 'Country', 'e.g. Rwanda'),
  FieldSpec('SurveyYear', 'Survey year', 'e.g. 2020', FieldType.integer),
  FieldSpec('CharacteristicCategory', 'Group type', 'e.g. Age (5-year groups)'),
  FieldSpec('CharacteristicLabel', 'Group', 'e.g. 20-24'),
  FieldSpec(
    'women_literate_pct',
    'Women who are literate (%)',
    '0 to 100',
    FieldType.decimal,
  ),
  FieldSpec(
    'women_secondary_edu_pct',
    'Women with some secondary education (%)',
    '0 to 100',
    FieldType.decimal,
  ),
  FieldSpec(
    'women_bank_account_pct',
    'Women with a bank account (%)',
    '0 to 100',
    FieldType.decimal,
  ),
  FieldSpec(
    'women_accept_wife_beating_pct',
    'Women who think wife beating is justified (%)',
    '0 to 100',
    FieldType.decimal,
  ),
  FieldSpec(
    'total_fertility_rate',
    'Total fertility rate (children per woman)',
    '0 to 10',
    FieldType.decimal,
  ),
];

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final Map<String, TextEditingController> _controllers = {
    for (final f in kFields) f.key: TextEditingController(),
  };
  final Map<String, String> _labels = {for (final f in kFields) f.key: f.label};

  bool _loading = false;
  String? _result;
  String? _error;

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  /// Builds the JSON body. Empty fields are left out, and text that is not a
  /// number is sent as text, so the API itself reports missing and wrong values.
  Map<String, dynamic> _buildBody() {
    final body = <String, dynamic>{};
    for (final f in kFields) {
      final text = _controllers[f.key]!.text.trim();
      if (text.isEmpty) continue;
      switch (f.type) {
        case FieldType.integer:
          body[f.key] = int.tryParse(text) ?? text;
          break;
        case FieldType.decimal:
          body[f.key] = double.tryParse(text) ?? text;
          break;
        case FieldType.text:
          body[f.key] = text;
          break;
      }
    }
    return body;
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    setState(() {
      _loading = true;
      _result = null;
      _error = null;
    });

    String? result;
    String? error;
    try {
      final response = await http
          .post(
            Uri.parse('$apiBaseUrl/predict'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(_buildBody()),
          )
          .timeout(const Duration(seconds: 90));
      final data = jsonDecode(response.body);
      if (response.statusCode == 200) {
        final value = (data['predicted_violence_pct'] as num).toStringAsFixed(
          2,
        );
        result =
            'Predicted: $value% of women in this group experienced '
            'physical and/or sexual partner violence in the last 12 months.';
      } else if (response.statusCode == 422) {
        error = _readableErrors(data['detail']);
      } else {
        error =
            'The server returned an error (code ${response.statusCode}). '
            'Please try again.';
      }
    } on TimeoutException {
      error =
          'The server took too long to answer. It may be waking up, '
          'so please try again in a moment.';
    } catch (_) {
      error =
          'Could not reach the server. Check your internet connection '
          'and try again.';
    }

    if (!mounted) return;
    setState(() {
      _loading = false;
      _result = result;
      _error = error;
    });
  }

  /// Turns the API's validation errors into short readable sentences.
  String _readableErrors(dynamic detail) {
    if (detail is String) return detail;
    if (detail is List) {
      final lines = <String>[];
      for (final item in detail) {
        if (item is! Map) continue;
        final loc = item['loc'];
        final key = (loc is List && loc.length > 1) ? loc.last.toString() : '';
        final label = _labels[key] ?? key;
        lines.add('$label: ${_friendly(item['type']?.toString() ?? '', item)}');
      }
      if (lines.isNotEmpty) return lines.join('\n');
    }
    return 'Please check your inputs and try again.';
  }

  String _friendly(String type, Map item) {
    final ctx = item['ctx'];
    final maxValue = ctx is Map ? ctx['le'] : null;
    final minValue = ctx is Map ? ctx['ge'] : null;
    switch (type) {
      case 'missing':
        return 'this value is required.';
      case 'int_parsing':
      case 'int_type':
      case 'int_from_float':
        return 'must be a whole number.';
      case 'float_parsing':
      case 'float_type':
        return 'must be a number.';
      case 'less_than_equal':
        return 'must be at most $maxValue.';
      case 'greater_than_equal':
        return 'must be at least $minValue.';
      case 'literal_error':
        return 'must be one of the 10 supported countries.';
      default:
        return '${item['msg'] ?? 'is not valid'}.';
    }
  }

  Widget _buildDisplay(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    String text;
    Color background;
    Color foreground;
    if (_loading) {
      text =
          'Contacting the server... the first request can take up to a '
          'minute while it wakes up.';
      background = scheme.surfaceContainerHighest;
      foreground = scheme.onSurface;
    } else if (_error != null) {
      text = _error!;
      background = scheme.errorContainer;
      foreground = scheme.onErrorContainer;
    } else if (_result != null) {
      text = _result!;
      background = scheme.primaryContainer;
      foreground = scheme.onPrimaryContainer;
    } else {
      text = 'Fill in the fields and tap Predict.';
      background = scheme.surfaceContainerHighest;
      foreground = scheme.onSurface;
    }
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(text, style: TextStyle(color: foreground, fontSize: 16)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('GBV Prevalence Predictor')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Estimate the % of women in a group who experienced partner '
                'violence in the last 12 months.',
                style: TextStyle(fontSize: 15),
              ),
              const SizedBox(height: 16),
              for (final f in kFields) ...[
                TextField(
                  controller: _controllers[f.key],
                  textInputAction: TextInputAction.next,
                  decoration: InputDecoration(
                    labelText: f.label,
                    hintText: f.hint,
                    border: const OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 12),
              ],
              const SizedBox(height: 4),
              FilledButton(
                onPressed: _loading ? null : _predict,
                child: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 14),
                  child: Text('Predict'),
                ),
              ),
              const SizedBox(height: 16),
              _buildDisplay(context),
            ],
          ),
        ),
      ),
    );
  }
}
