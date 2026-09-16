import sys

with open('src/components/AlertManager.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '<div className="flex-1 flex flex-col min-h-0">',
    '<div className="flex-1 flex flex-col min-h-0 overflow-y-auto">'
)

content = content.replace(
    '<div className="p-6 flex-1 flex flex-col space-y-4 overflow-y-auto border-b border-slate-100">',
    '<div className="p-6 flex flex-col space-y-4 border-b border-slate-100">'
)

content = content.replace(
    '<div className="p-6 flex-1 bg-slate-900 text-white overflow-y-auto">',
    '<div className="p-6 bg-slate-900 text-white">'
)

with open('src/components/AlertManager.jsx', 'w', encoding='utf-8') as f:
    f.write(content)

print('Update successful')
